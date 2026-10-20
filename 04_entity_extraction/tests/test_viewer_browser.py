"""Remote launch requests must not block or crash the viewer."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess, TimeoutExpired

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewer_server import open_viewer_browser


class BrowserLaunchTests(unittest.TestCase):
    @patch('viewer_server.subprocess.run')
    @patch('viewer_server.shutil.which', return_value='/bin/code')
    @patch.dict('os.environ', {'VSCODE_IPC_HOOK_CLI': '/tmp/test-ipc'})
    def test_remote_launch_uses_client_and_localhost(self, which, run):
        run.return_value = CompletedProcess([], 0, '', '')
        open_viewer_browser('127.0.0.1', 5006)
        self.assertEqual(run.call_args.args[0],
                         ['/bin/code', '--openExternal', 'http://localhost:5006'])
        self.assertEqual(run.call_args.kwargs['timeout'], 20)

    @patch('viewer_server.subprocess.run')
    @patch('viewer_server.shutil.which', return_value=None)
    def test_no_client_does_not_launch_remote_browser(self, which, run):
        open_viewer_browser('127.0.0.1', 5006)
        run.assert_not_called()

    @patch('viewer_server.subprocess.run', side_effect=TimeoutExpired('code', 20))
    @patch('viewer_server.shutil.which', return_value='/bin/code')
    @patch.dict('os.environ', {'VSCODE_IPC_HOOK_CLI': '/tmp/test-ipc'})
    def test_stalled_client_does_not_raise(self, which, run):
        open_viewer_browser('127.0.0.1', 5006)


if __name__ == '__main__':
    unittest.main()
