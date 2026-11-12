import unittest

from forecasting_tools import GeneralLlm


class CredentialSerializationTests(unittest.TestCase):
    def test_credentials_are_not_serialized_into_model_configuration(self):
        key = "synthetic-audit-key-never-valid-" + "a" * 40
        header = "Bearer synthetic-audit-header-" + "b" * 40
        model = GeneralLlm(model="openai/gpt-4o", api_key=key,
                           extra_headers={"Authorization": header}, temperature=0.25)
        serialized = str(model.to_dict())
        self.assertNotIn(key, serialized)
        self.assertNotIn(header, serialized)
        self.assertIn("temperature", serialized)
