"""Human GT must keep source coordinates, uncertainty and explicit review boundaries."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from de14.annotations import validate_labels, vehicle_violation


class VideoAnnotationsTest(unittest.TestCase):
    def fixture(self):
        vehicle={'id':'V001','box':[10,10,190,190],'occupancy':'occupied','rider_count':2,
                 'heads':[{'id':'R01','box':[20,20,40,40],'helmet':'helmet'},
                          {'id':'R02','box':[60,20,80,40],'helmet':'unknown'}]}
        return {'schema_version':1,'source':{'sha256':'a'*64,'fps':30,'width':200,'height':200,'frame_count':60},
                'scope':'sampled_source_frames','frames':[{'source_index':0,'timestamp_s':0,'reviewed':False,'vehicles':[vehicle]},
                                                       {'source_index':10,'timestamp_s':1/3,'reviewed':False,'vehicles':[]}]}

    def test_partial_labels_do_not_become_reviewed_ground_truth(self):
        data=self.fixture();stats=validate_labels(data,expected_source_sha256='a'*64)
        self.assertEqual(stats['reviewed_frames'],0)
        self.assertEqual(stats['reviewed_vehicles'],0)
        data['frames'][0]['reviewed']=True
        self.assertEqual(validate_labels(data)['reviewed_vehicles'],1)

    def test_unknown_or_occluded_riders_are_not_all_helmeted(self):
        vehicle=self.fixture()['frames'][0]['vehicles'][0]
        self.assertIsNone(vehicle_violation(vehicle))
        vehicle['heads'][1]['helmet']='helmet'
        self.assertIs(vehicle_violation(vehicle),False)
        vehicle['rider_count']=3
        self.assertIsNone(vehicle_violation(vehicle))
        vehicle['heads'][0]['helmet']='no_helmet'
        self.assertIs(vehicle_violation(vehicle),True)
        vehicle['occupancy']='unknown'
        self.assertIsNone(vehicle_violation(vehicle))

    def test_source_mismatch_cannot_mix_different_videos(self):
        with self.assertRaisesRegex(ValueError,'source'):
            validate_labels(self.fixture(),expected_source_sha256='b'*64)

    def test_vehicle_id_can_repeat_across_frames_but_not_twice_in_one_frame(self):
        data=self.fixture();vehicle=data['frames'][0]['vehicles'][0]
        data['frames'][1]['vehicles']=[copy.deepcopy(vehicle)]
        validate_labels(data)
        data['frames'][0]['vehicles'].append(copy.deepcopy(vehicle))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_labels(data)

    def test_boxes_are_validated_in_original_source_pixels(self):
        data=self.fixture();data['frames'][0]['vehicles'][0]['heads'][0]['box']=[10,10,201,30]
        with self.assertRaisesRegex(ValueError,'box'):
            validate_labels(data)

    def test_empty_motorcycle_cannot_have_rider_heads(self):
        data=self.fixture();data['frames'][0]['vehicles'][0]['occupancy']='empty'
        with self.assertRaisesRegex(ValueError,'empty'):
            validate_labels(data)

    def test_timestamp_must_match_source_frame_and_fps(self):
        data=self.fixture();data['frames'][1]['timestamp_s']=10
        with self.assertRaisesRegex(ValueError,'timestamp'):
            validate_labels(data)


if __name__=='__main__': unittest.main()
