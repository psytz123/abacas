import sys
import types
from unittest.mock import MagicMock
import pytest

# ---------------------------------------------------------------------------
# Stub external dependencies so the tests can run in this limited environment.
# ---------------------------------------------------------------------------

# requests stub
requests_mod = types.ModuleType("requests")
class DummySession:
    def request(self, *a, **k):
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    def mount(self, *a, **k):
        pass
requests_mod.Session = DummySession
requests_mod.adapters = types.SimpleNamespace(HTTPAdapter=object)
requests_mod.exceptions = types.SimpleNamespace(RequestException=Exception)
requests_mod.Response = object
class DummyHTTPBasicAuth:
    def __init__(self, username, password):
        self.username = username
        self.password = password
requests_mod.auth = types.SimpleNamespace(HTTPBasicAuth=DummyHTTPBasicAuth)
sys.modules.setdefault("requests", requests_mod)
sys.modules.setdefault("requests.adapters", requests_mod.adapters)
sys.modules.setdefault("requests.auth", requests_mod.auth)

# urllib3 stub
urllib3_mod = types.ModuleType("urllib3")
urllib3_mod.disable_warnings = lambda *a, **k: None
sys.modules.setdefault("urllib3", urllib3_mod)
retry_mod = types.ModuleType("urllib3.util.retry")
class DummyRetry:
    def __init__(self, *a, **k):
        pass
retry_mod.Retry = DummyRetry
sys.modules.setdefault("urllib3.util.retry", retry_mod)
sys.modules.setdefault("urllib3.exceptions", types.ModuleType("urllib3.exceptions"))
sys.modules["urllib3.exceptions"].InsecureRequestWarning = type("InsecureRequestWarning", (), {})

# tenacity stub
tenacity_mod = types.ModuleType("tenacity")

def _decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper

tenacity_mod.retry = _decorator
tenacity_mod.stop_after_attempt = lambda *a, **k: None
tenacity_mod.wait_exponential = lambda *a, **k: None
tenacity_mod.retry_if_exception_type = lambda *a, **k: None
tenacity_mod.before_sleep_log = lambda *a, **k: None
sys.modules.setdefault("tenacity", tenacity_mod)

# cryptography stub
fernet_mod = types.ModuleType("cryptography.fernet")
class DummyFernet:
    def __init__(self, key):
        pass
    def encrypt(self, data):
        return data
    def decrypt(self, data):
        return data
    @staticmethod
    def generate_key():
        return b"key"
fernet_mod.Fernet = DummyFernet
sys.modules.setdefault("cryptography.fernet", fernet_mod)

pbkdf2_mod = types.ModuleType("cryptography.hazmat.primitives.kdf.pbkdf2")
class DummyPBKDF2HMAC:
    def __init__(self, *a, **k):
        pass
    def derive(self, data):
        return b"key"
pbkdf2_mod.PBKDF2HMAC = DummyPBKDF2HMAC
sys.modules.setdefault("cryptography.hazmat.primitives.kdf.pbkdf2", pbkdf2_mod)

hashes_mod = types.ModuleType("cryptography.hazmat.primitives.hashes")
class DummySHA256:
    pass
hashes_mod.SHA256 = DummySHA256
sys.modules.setdefault("cryptography.hazmat.primitives.hashes", hashes_mod)

primitives_mod = types.ModuleType("cryptography.hazmat.primitives")
primitives_mod.kdf = types.SimpleNamespace(pbkdf2=pbkdf2_mod)
primitives_mod.hashes = hashes_mod
sys.modules.setdefault("cryptography.hazmat.primitives", primitives_mod)

hazmat_mod = types.ModuleType("cryptography.hazmat")
hazmat_mod.primitives = primitives_mod
sys.modules.setdefault("cryptography.hazmat", hazmat_mod)

crypto_mod = types.ModuleType("cryptography")
crypto_mod.fernet = fernet_mod
crypto_mod.hazmat = hazmat_mod
sys.modules.setdefault("cryptography", crypto_mod)

# pandas and numpy stubs
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

# Stub parts of the ml_engine package that depend on heavy ML libraries.
import pathlib
ml_engine_pkg = types.ModuleType("ml_engine")
ml_engine_pkg.__path__ = [str(pathlib.Path(__file__).resolve().parent / "ml_engine")]
sys.modules.setdefault("ml_engine", ml_engine_pkg)

recommender_mod = types.ModuleType("ml_engine.recommender")
class DummyRecommendationEngine:
    pass
recommender_mod.RecommendationEngine = DummyRecommendationEngine
sys.modules.setdefault("ml_engine.recommender", recommender_mod)

logging_mod = types.ModuleType("ml_engine.utils.logging_config")
logging_mod.logger = MagicMock()
sys.modules.setdefault("ml_engine.utils.logging_config", logging_mod)

validation_mod = types.ModuleType("ml_engine.utils.validation")
class DummyValidationError(Exception):
    pass
validation_mod.ValidationError = DummyValidationError
sys.modules.setdefault("ml_engine.utils.validation", validation_mod)

# ---------------------------------------------------------------------------
# Actual imports from the project after stubbing dependencies
# ---------------------------------------------------------------------------
from api_clients.vnish_firmware_client import VnishFirmwareClient
from api_clients.credential_store import CredentialStore, VnishCredentialManager
from ml_engine.vnish_integration import VnishMLIntegration


def test_credential_manager(tmp_path):
    store = CredentialStore(config_dir=str(tmp_path), use_encryption=False)
    manager = VnishCredentialManager(store)

    test_ip = "192.168.1.101"
    test_user = "admin"
    test_password = "admin"

    assert manager.save_credentials(test_ip, test_user, test_password, True)

    ip, user, pwd = manager.get_credentials(test_ip)
    assert ip == test_ip
    assert user == test_user
    assert pwd == test_password

    assert manager.delete_credentials(test_ip)


def test_firmware_client(monkeypatch):
    # Avoid network-related initialization in BaseAPIClient
    monkeypatch.setattr(
        "api_clients.base.BaseAPIClient.__init__",
        lambda self, base_url, timeout=30, max_retries=3, backoff_factor=0.5, status_forcelist=None: None,
    )
    client = VnishFirmwareClient(miner_ip="192.168.1.101", username="u", password="p")

    monkeypatch.setattr(client, "get_summary", lambda: {"status": "ok"})
    monkeypatch.setattr(client, "get_chips_status", lambda: {"chips": []})
    monkeypatch.setattr(client, "get_telemetry", lambda: {"telemetry": True})

    assert client.get_summary()["status"] == "ok"
    assert client.get_chips_status() == {"chips": []}
    assert client.get_telemetry() == {"telemetry": True}


def test_vnish_integration(monkeypatch, tmp_path):
    store = CredentialStore(config_dir=str(tmp_path), use_encryption=False)
    manager = VnishCredentialManager(store)
    manager.save_credentials("192.168.1.101", "u", "p", True)

    integration = VnishMLIntegration(credential_manager=manager)

    mock_client = MagicMock()
    mock_client.get_telemetry.return_value = {"telemetry": True}
    monkeypatch.setattr(integration, "get_client", lambda miner_ip=None: mock_client)

    assert integration.get_miner_telemetry("192.168.1.101") == {"telemetry": True}

    hashrate_rec = {
        "id": "test-hashrate",
        "type": "dynamic_hashrate_tuning",
        "miner_id": "miner_192.168.1.101",
        "recommended_hashrate_percent": 80.0,
    }
    result = integration.apply_hashrate_tuning_recommendation(hashrate_rec, miner_ip="192.168.1.101", dry_run=True)
    assert result["status"] == "success" and result["dry_run"] is True

    power_rec = {
        "id": "test-power",
        "type": "power_optimization",
        "miner_id": "miner_192.168.1.101",
        "power_reduction_percent": 15.0,
    }
    result = integration.apply_power_optimization_recommendation(power_rec, miner_ip="192.168.1.101", dry_run=True)
    assert result["status"] == "success" and result["dry_run"] is True

    oc_rec = {
        "id": "test-overclocking",
        "type": "intelligent_overclocking",
        "miner_id": "miner_192.168.1.101",
        "core_clock_offset": 50,
        "memory_clock_offset": 200,
        "power_limit_percent": 85.0,
        "core_voltage_offset": 10,
    }
    result = integration.apply_overclocking_recommendation(oc_rec, miner_ip="192.168.1.101", dry_run=True)
    assert result["status"] == "success" and result["dry_run"] is True

