"""Real ByteTrack: restore measured history, never turn predictions into votes."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from de14.pipeline import Pipeline
from de14.core import UNDET, VIOL

EMPTY={'motorcycle':[],'motorcycle_score':[],'person':[],'head':[],'head_detection':True}
SCENE={'motorcycle':[[20,70,180,190]],'motorcycle_score':[.9],'person':[],
       'head':[[40,10,70,40]],'head_detection':True}

class TentativeObservationsTest(unittest.TestCase):
    def run_scene(self, scenes, values=None, **kwargs):
        detector=iter(scenes);calls=[]
        def classifier(crops):
            calls.append([int(c[0,0,0]) for c in crops])
            return [.9 if c[0,0,0]<100 else .1 for c in crops]
        frames=[(i,np.full((200,200,3),v,np.uint8)) for i,v in enumerate(values or [11]*len(scenes))]
        meta={i:{'source_index':i*10,'timestamp_s':i/3} for i in range(len(frames))}
        result=Pipeline(lambda _:next(detector),classifier,n_min=2,batch_heads=True,**kwargs).run(
            frames,write_video=False,evidence=False,frame_metadata=meta)
        return result,calls

    def test_confirmation_recovers_the_original_frame_crop_and_source_timestamp(self):
        result,calls=self.run_scene([EMPTY,SCENE,SCENE],[0,11,200])
        self.assertEqual(calls,[[11,200]])
        tid=next(iter(result['tracks']))
        self.assertEqual([o['source_index'] for o in result['observations'][tid]],[10,20])
        self.assertEqual([o['t'] for o in result['observations'][tid]],[1/3,2/3])
        self.assertEqual([o['p_no_helmet'] for o in result['observations'][tid]],[.9,.1])
        self.assertEqual(len(result['per_frame'][1][1]),1)
        self.assertEqual(result['frame_decisions'][1][tid],UNDET)
        self.assertEqual(result['frame_decisions'][2][tid],UNDET)
        self.assertEqual(result['temporal_backfill']['vehicle_frames'],1)

    def test_actual_past_observation_can_complete_vote_count_without_future_vote(self):
        result,_=self.run_scene([EMPTY,SCENE,SCENE])
        tid=next(iter(result['tracks']))
        self.assertEqual(result['tracks'][tid]['n_obs'],2)
        self.assertEqual(result['frame_decisions'][1][tid],UNDET)
        self.assertEqual(result['frame_decisions'][2][tid],VIOL)

    def test_never_confirmed_candidate_is_not_classified_or_reported_as_violation(self):
        result,calls=self.run_scene([EMPTY,SCENE])
        self.assertEqual(calls,[])
        self.assertEqual(sum(map(len,result['observations'].values())),0)
        self.assertEqual(result['tracks'],{})
        self.assertEqual(result['temporal_backfill']['discarded_vehicle_frames'],1)

    def test_removed_candidate_does_not_leak_into_a_later_identity(self):
        result,calls=self.run_scene([EMPTY,SCENE,EMPTY,SCENE,SCENE],[0,7,0,22,33])
        self.assertEqual(calls,[[22,33]])
        obs=next(iter(result['observations'].values()))
        self.assertEqual([o['source_index'] for o in obs],[30,40])
        self.assertEqual(result['temporal_backfill']['discarded_vehicle_frames'],1)

    def test_legacy_mode_still_has_only_the_confirmation_frame(self):
        result,calls=self.run_scene([EMPTY,SCENE,SCENE],backfill_tentative=False)
        self.assertEqual(calls,[[11]])
        self.assertEqual([len(r) for _,r in result['per_frame']],[0,0,1])

    def test_buffered_crop_survives_decoder_reusing_its_frame_array(self):
        scenes=iter([EMPTY,SCENE,SCENE]);seen=[];image=np.zeros((200,200,3),np.uint8)
        def frames():
            for i,value in enumerate([0,11,200]):
                image[:]=value
                yield i,image
        def cls(crops):
            seen.extend(int(c[0,0,0]) for c in crops)
            return [.9]*len(crops)
        Pipeline(lambda _:next(scenes),cls,batch_heads=True).run(frames(),write_video=False,evidence=False)
        self.assertEqual(seen,[11,200])

    def test_two_heads_keep_unknown_and_one_vote_per_measured_vehicle_frame(self):
        scene={**SCENE,'head':[[30,10,60,40],[90,20,120,50]]}
        scenes=iter([EMPTY,scene,scene]);seen=[]
        def cls(crops):
            seen.append(len(crops));return [.9,None,.1,.1]
        result=Pipeline(lambda _:next(scenes),cls,batch_heads=True,n_min=2).run(
            [(i,np.zeros((200,200,3),np.uint8)) for i in range(3)],write_video=False,evidence=False)
        tid=next(iter(result['tracks']))
        self.assertEqual(seen,[4])
        self.assertEqual([p for _,p in result['per_frame'][1][1][0][3]],[.9,None])
        self.assertEqual([p for _,p in result['per_frame'][2][1][0][3]],[.1,.1])
        self.assertEqual(result['tracks'][tid]['n_obs'],2)
        self.assertEqual(result['counters']['associated_heads'],4)

    def test_streaming_render_does_not_claim_to_rewrite_prior_frames(self):
        scenes=iter([EMPTY,SCENE,SCENE]);calls=[]
        def cls(crops):calls.append(len(crops));return [.9]*len(crops)
        result=Pipeline(lambda _:next(scenes),cls).run(
            [(i,np.zeros((200,200,3),np.uint8)) for i in range(3)],write_video=False,evidence=False,render=True)
        self.assertFalse(result['temporal_backfill']['enabled'])
        self.assertEqual(calls,[1])
        self.assertEqual([len(rs) for _,rs in result['per_frame']],[0,0,1])

    def test_trace_separates_observed_frame_from_later_processing_frame(self):
        scenes=iter([EMPTY,SCENE,SCENE]);events=[]
        result=Pipeline(lambda _:next(scenes),lambda cs:[.9]*len(cs),batch_heads=True).run(
            [(i,np.zeros((200,200,3),np.uint8)) for i in range(3)],write_video=False,evidence=False,trace=events.append)
        restored=[e for e in events if e['stage']=='classification' and e['frame']==1 and e.get('processing_frame')==2]
        self.assertEqual(len(restored),1)
        self.assertEqual(len(restored[0]['heads']),1)
        self.assertEqual(result['temporal_backfill']['events'][0]['confirmed_at_source_index'],2)

    def test_restored_head_evidence_keeps_its_own_detector_mode_for_f2(self):
        scenes=iter([EMPTY,{k:v for k,v in SCENE.items() if k!='head_detection'},
                     {**SCENE,'head':[],'head_detection':False}])
        result=Pipeline(lambda _:next(scenes),lambda cs:[.9]*len(cs),batch_heads=True,
                        track_filter={'tau_disp':.75,'tau_person_static':.4}).run(
            [(i,np.zeros((200,200,3),np.uint8)) for i in range(3)],write_video=False,evidence=False)
        audit=next(iter(result['track_audit'].values()))
        self.assertEqual(audit['head_positive_frames'],1)
        self.assertEqual(audit['person_score'],.5)
        self.assertTrue(audit['kept'])

    def test_public_benchmark_scores_actual_temporal_output(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
        from benchmark_video_tracking import predictions
        result={'per_frame':[(i,[(1,[0,0,100,100],p,[([10,10,30,30],p)])]) for i,p in enumerate([.9,.1,.1])],
                'frame_decisions':{0:{1:VIOL},1:{1:UNDET},2:{1:'có đội mũ'}}}
        self.assertEqual([f['vehicles'][0]['decision'] for f in predictions(result,'clip')],[VIOL,UNDET,'có đội mũ'])

    def test_headless_tentative_frame_does_not_add_unsupported_past_vehicle_box(self):
        no_heads={**SCENE,'head':[]}
        result,calls=self.run_scene([EMPTY,no_heads,SCENE])
        self.assertEqual([len(rs) for _,rs in result['per_frame']],[0,0,1])
        self.assertEqual(calls,[[11]])
        self.assertEqual(result['temporal_backfill']['vehicle_frames'],0)

    def test_only_invalid_tentative_crops_do_not_add_a_past_vehicle_box(self):
        small={**SCENE,'head':[[40,10,41,11]]}
        result,_=self.run_scene([EMPTY,small,SCENE])
        self.assertEqual([len(rs) for _,rs in result['per_frame']],[0,0,1])

    def test_valid_unknown_tentative_crop_is_preserved(self):
        scenes=iter([EMPTY,SCENE,SCENE])
        result=Pipeline(lambda _:next(scenes),lambda cs:[None]*len(cs),batch_heads=True).run(
            [(i,np.zeros((200,200,3),np.uint8)) for i in range(3)],write_video=False,evidence=False)
        self.assertEqual([len(rs) for _,rs in result['per_frame']],[0,1,1])
        self.assertEqual(sum(map(len,result['observations'].values())),0)
        self.assertEqual(result['temporal_backfill']['vehicle_frames'],1)

if __name__=='__main__':unittest.main()
