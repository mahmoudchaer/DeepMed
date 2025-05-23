import pytest
from unittest.mock import patch, MagicMock
from werkzeug.datastructures import FileStorage
from io import BytesIO
import app_images
import app_api

@pytest.fixture
def app():
    app_images.app.config['TESTING'] = True
    app_images.app.secret_key = 'test_secret_key'
    return app_images.app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    user.is_authenticated = True
    return user

@pytest.fixture(autouse=True)
def patch_user_loader(monkeypatch, mock_user):
    monkeypatch.setattr(app_api, 'load_user', lambda user_id: mock_user)

# --- /images redirect ---
def test_images_route_redirect(client, mock_user):
    with patch('app_images.current_user', mock_user):
        with client.session_transaction() as sess:
            sess['_user_id'] = mock_user.id
        resp = client.get('/images')
        assert resp.status_code == 302
        assert '/pipeline' in resp.location

# --- unauthenticated test is fine ---
def test_anomaly_detection_route_unauthenticated(client):
    unauth_user = MagicMock()
    unauth_user.is_authenticated = False
    with patch('app_images.current_user', unauth_user):
        with client.session_transaction() as sess:
            sess.pop('_user_id', None)
        resp = client.get('/anomaly_detection', follow_redirects=False)
        assert resp.status_code == 302
        assert '/login' in resp.location

# --- /api/train_model ---
def test_api_train_model_success(client, mock_user):
    with patch('app_images.current_user', mock_user), \
         patch('app_images.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'modeldata'
        mock_response.headers = {'X-Training-Metrics': '{}'}
        mock_post.return_value = mock_response
        with client.session_transaction() as sess:
            sess['_user_id'] = mock_user.id
        data = {
            'zipFile': (BytesIO(b'zipcontent'), 'test.zip'),
            'numClasses': '2',
            'trainingLevel': '1'
        }
        resp = client.post('/api/train_model', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert resp.headers['Content-Type'] == 'application/octet-stream'
        assert b'modeldata' in resp.data

def test_api_train_model_failure(client, mock_user):
    with patch('app_images.current_user', mock_user), \
         patch('app_images.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'error': 'fail'}
        mock_post.return_value = mock_response
        with client.session_transaction() as sess:
            sess['_user_id'] = mock_user.id
        data = {
            'zipFile': (BytesIO(b'zipcontent'), 'test.zip'),
            'numClasses': '2',
            'trainingLevel': '1'
        }
        resp = client.post('/api/train_model', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
        assert b'fail' in resp.data

def test_process_augmentation_success(client, mock_user):
    with patch('app_images.current_user', mock_user), \
         patch('app_images.is_service_available', return_value=True), \
         patch('app_images.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'augzip'
        mock_post.return_value = mock_response
        with client.session_transaction() as sess:
            sess['_user_id'] = mock_user.id
        data = {
            'zipFile': (BytesIO(b'zipcontent'), 'test.zip'),
            'level': '2'
        }
        resp = client.post('/augment/process', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert resp.headers['Content-Type'] == 'application/zip'
        assert b'augzip' in resp.data

def test_process_augmentation_failure(client, mock_user):
    with patch('app_images.current_user', mock_user), \
         patch('app_images.is_service_available', return_value=True), \
         patch('app_images.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'error': 'bad'}
        mock_post.return_value = mock_response
        with client.session_transaction() as sess:
            sess['_user_id'] = mock_user.id
        data = {
            'zipFile': (BytesIO(b'zipcontent'), 'test.zip'),
            'level': '2'
        }
        resp = client.post('/augment/process', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
        assert b'bad' in resp.data 