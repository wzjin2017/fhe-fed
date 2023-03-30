import pyhelayers
import torch
import pickle
import os

input = torch.rand(5609)

response_text = {"requirement": {"batchSize": 16, "security_level": "128", "integer_part_precision": "10", "fractional_part_precision": "50", "num_slots": "8192", "multiplication_depth": "6"}, "tile_layout": {"num_dims": "3", "t1": "4", "t2": "128", "t3": "16"}, "mode": "DEFAULT", "model_encrypted": "true", "lazy_encoding": "false", "measures": {"predict_cpu_time": "381413", "init_model_cpu_time": "161777", "encrypt_input_cpu_time": "145224", "decrypt_output_cpu_time": "123", "model_memory": "16254017", "input_memory": "14681032", "output_memory": "262265", "context_memory": "61786770", "client_latency": "-1", "server_latency": "-1", "client_upload_time": "-1", "server_upload_time": "-1", "latency": "-1", "throughput": "41.949278079142559", "chain_index_consumed": "6", "bootstraps": "0"}}
profile = pyhelayers.HeProfile()
profile.from_string(response_text)

client_context = pyhelayers.DefaultContext()
client_context.init(profile.requirement)


loaded_iop=pyhelayers.ModelIoProcessor(client_context)
encrypted_data_samples = loaded_iop.encode_encrypt_input(input)

with open('ciphertext_seal.pickle', 'wb') as handle:
    pickle.dump(encrypted_data_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
cipher_size = os.path.getsize('ciphertext_helayers.pickle')
print(cipher_size)