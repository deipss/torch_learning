import json

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


def batch_translate():
    k_access_key = 'AKLTNzYzMDhlY2Y3OGJmNDA2NmIzMWViYWExMDBjMDJhNDc'  # https://console.volcengine.com/iam/keymanage/
    k_secret_key = 'TW1Wa04yRTNNR1ZoWkRnME5EQmxPVGd5TkdNeU16aGlPVFpoTWpFeE9XWQ=='
    k_service_info = \
        ServiceInfo('translate.volcengineapi.com',
                    {'Content-Type': 'application/json'},
                    Credentials(k_access_key, k_secret_key, 'translate', 'cn-north-1'),
                    5,
                    5)
    k_query = {
        'Action': 'TranslateText',
        'Version': '2020-06-01'
    }
    k_api_info = {
        'translate': ApiInfo('POST', '/', k_query, {}, {})
    }
    service = Service(k_service_info, k_api_info)
    body = {
        'TargetLanguage': 'en',
        'TextList': ["⑴投运后，流向、温升和声响正常，无渗漏；", "⑵强油水冷装置的检查和试验，按制造厂规定；"],
    }
    res = service.json('translate', {}, json.dumps(body))
    print(json.loads(res))


if __name__ == '__main__':
    batch_translate()
