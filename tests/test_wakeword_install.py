import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import install_wakeword as installer


class InstallTests(unittest.TestCase):
    def archive(self):
        model, url, _, files = installer.PROFILES['english']
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w:bz2') as archive:
            for name in (*files, '../ignored'):
                member = tarfile.TarInfo(f'{model}/{name}')
                member.size = 5
                archive.addfile(member, io.BytesIO(b'model'))
        payload = data.getvalue()
        return payload, (model, url, hashlib.sha256(payload).hexdigest(), files)

    def test_english_installs_only_verified_allowlisted_members(self):
        payload, profile = self.archive()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'english'
            with patch.dict(installer.PROFILES, english=profile), patch.object(
                installer.urllib.request, 'urlopen', return_value=io.BytesIO(payload)
            ) as download:
                installer.install(destination, 'english')
            download.assert_called_once_with(profile[1], timeout=60)
            self.assertEqual({p.name for p in destination.iterdir()}, set(profile[3]))
            self.assertFalse((Path(directory) / 'ignored').exists())

    def test_bad_checksum_leaves_destination_untouched(self):
        payload, _ = self.archive()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'english'
            with patch.object(installer.urllib.request, 'urlopen',
                              return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    installer.install(destination, 'english')
            self.assertFalse(destination.exists())

    def test_unknown_profile_never_downloads(self):
        with patch.object(installer.urllib.request, 'urlopen') as download:
            with self.assertRaises(ValueError):
                installer.install(Path('unused'), 'unknown')
            download.assert_not_called()
