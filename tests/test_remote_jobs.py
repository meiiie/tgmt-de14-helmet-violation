"""Portable inference jobs must preserve media/options and reject unsafe archives."""
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


class RemoteArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_request_round_trip_preserves_input_and_inference_options(self):
        from de14.remote_jobs import JobOptions, create_request, read_request
        media = self.root/'input.mp4'; media.write_bytes(b'original media bytes')
        options = JobOptions(kind='video', detector='sam3.1', accurate=False, stride=5, n_min=10)
        archive = self.root/'request.zip'
        created = create_request(media, options, archive)
        manifest, extracted = read_request(archive, self.root/'extracted')
        self.assertEqual(extracted.read_bytes(), b'original media bytes')
        self.assertEqual(manifest['job_id'], created['job_id'])
        self.assertEqual(manifest['options']['detector'], 'sam3.1')
        self.assertEqual(manifest['options']['stride'], 5)
        self.assertEqual(manifest['options']['n_min'], 10)
        self.assertFalse(manifest['options']['accurate'])
        self.assertTrue(manifest['options']['detect_heads'])

    def test_tampered_media_is_not_used(self):
        from de14.remote_jobs import JobOptions, create_request, read_request
        media = self.root/'input.png'; media.write_bytes(b'original')
        good = self.root/'request.zip'; create_request(media, JobOptions(kind='image'), good)
        bad = self.root/'tampered.zip'
        with ZipFile(good) as source, ZipFile(bad, 'w') as target:
            for entry in source.infolist():
                target.writestr(entry.filename, b'tampered' if entry.filename.startswith('input/') else source.read(entry))
        with self.assertRaisesRegex(ValueError, 'checksum|size'):
            read_request(bad, self.root/'target')

    def test_traversal_and_duplicate_archive_entries_are_rejected(self):
        from de14.remote_jobs import read_request
        for name in ('../escape.txt', '/absolute.txt', 'C:/escape.txt', 'input\\escape.txt'):
            archive = self.root/'unsafe.zip'
            with ZipFile(archive, 'w') as target: target.writestr(name, b'no')
            with self.assertRaises(ValueError): read_request(archive, self.root/'target')
        archive = self.root/'duplicate.zip'
        import warnings
        with ZipFile(archive,'w') as target, warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            target.writestr('request.json', '{}'); target.writestr('request.json', '{}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_request(archive, self.root/'target')
        self.assertFalse((self.root/'escape.txt').exists())

    def test_options_cannot_select_code_paths_or_invalid_sampling(self):
        from de14.remote_jobs import JobOptions
        for values in ({'stride':0}, {'n_min':0}, {'detector':'arbitrary.py'}, {'output_mode':'other'}, {'imgsz':1}):
            with self.assertRaises((ValueError, TypeError)): JobOptions(kind='video', **values)

    def test_result_round_trip_checks_job_identity_and_artifact_checksum(self):
        from de14.remote_jobs import create_result, read_result
        artifact = self.root/'tracks.csv'; artifact.write_bytes(b'track_id,n_obs\n1,10\n')
        output = self.root/'result.zip'; job = 'a'*32
        create_result(job,'b'*64,{'assets':{'csv':'tracks.csv'}},{'tracks.csv':artifact},output)
        with self.assertRaisesRegex(ValueError,'different job'):
            read_result(output,self.root/'wrong',expected_job_id='c'*32)
        manifest, files = read_result(output,self.root/'correct',expected_job_id=job,expected_input_sha256='b'*64)
        self.assertEqual((files/'tracks.csv').read_bytes(),b'track_id,n_obs\n1,10\n')
        self.assertEqual(manifest['result']['assets']['csv'],'tracks.csv')

    def test_failed_packaging_does_not_leave_a_downloadable_partial_job(self):
        from de14.remote_jobs import create_result
        output = self.root/'failed.zip'
        with self.assertRaises(FileNotFoundError):
            create_result('a'*32,'b'*64,{}, {'missing.csv':self.root/'missing.csv'},output)
        self.assertFalse(output.exists())

    def test_packaging_does_not_overwrite_existing_result(self):
        from de14.remote_jobs import create_result
        output = self.root/'existing.zip'; output.write_bytes(b'keep this')
        with self.assertRaises(FileExistsError): create_result('a'*32,'b'*64,{}, {},output)
        self.assertEqual(output.read_bytes(),b'keep this')


if __name__ == '__main__': unittest.main()
