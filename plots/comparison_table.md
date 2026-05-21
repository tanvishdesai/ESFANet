# Final Performance Comparison Table

## Standard Threshold (0.50)

| Model | Scenario | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|----------|-----------|--------|----------|-----|
| Consensus Only | A | 0.8484 | 0.8553 | 0.8384 | 0.8468 | 0.9262 |
| Consensus Only | B | 0.8203 | 0.7856 | 0.8807 | 0.8305 | 0.8986 |
| Consensus Only | C | 0.8303 | 0.8287 | 0.8325 | 0.8306 | 0.9134 |
| FedAlert Only | A | 0.8169 | 0.7676 | 0.9088 | 0.8322 | 0.9170 |
| FedAlert Only | B | 0.8123 | 0.9240 | 0.6803 | 0.7837 | 0.9188 |
| FedAlert Only | C | 0.4998 | 0.4998 | 1.0000 | 0.6665 | 0.8740 |
| FedAvg | A | 0.8275 | 0.8135 | 0.8498 | 0.8312 | 0.9073 |
| FedAvg | B | 0.8353 | 0.8440 | 0.8226 | 0.8331 | 0.9187 |
| FedAvg | C | 0.8085 | 0.7865 | 0.8467 | 0.8155 | 0.8975 |
| FedProx | A | 0.8337 | 0.8436 | 0.8190 | 0.8311 | 0.9188 |
| FedProx | B | 0.8316 | 0.8321 | 0.8307 | 0.8314 | 0.9146 |
| FedProx | C | 0.8226 | 0.8189 | 0.8283 | 0.8236 | 0.9093 |
| Full (FedAlert + Consensus) | A | 0.8032 | 0.9200 | 0.6639 | 0.7712 | 0.9201 |
| Full (FedAlert + Consensus) | B | 0.8179 | 0.9066 | 0.7086 | 0.7954 | 0.9197 |
| Full (FedAlert + Consensus) | C | 0.8076 | 0.7619 | 0.8948 | 0.8230 | 0.9052 |

## FedAlert Threshold (0.75)

| Model | Scenario | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|----------|-----------|--------|----------|-----|
| Consensus Only | A | 0.8255 | 0.9095 | 0.7227 | 0.8054 | 0.9262 |
| Consensus Only | B | 0.8182 | 0.8503 | 0.7722 | 0.8093 | 0.8986 |
| Consensus Only | C | 0.8197 | 0.8914 | 0.7280 | 0.8015 | 0.9134 |
| FedAlert Only | A | 0.8051 | 0.9193 | 0.6688 | 0.7743 | 0.9170 |
| FedAlert Only | B | 0.7661 | 0.9515 | 0.5606 | 0.7055 | 0.9188 |
| FedAlert Only | C | 0.4998 | 0.4998 | 1.0000 | 0.6665 | 0.8740 |
| FedAvg | A | 0.8221 | 0.8760 | 0.7503 | 0.8083 | 0.9073 |
| FedAvg | B | 0.8116 | 0.9089 | 0.6924 | 0.7860 | 0.9187 |
| FedAvg | C | 0.8081 | 0.8545 | 0.7424 | 0.7945 | 0.8975 |
| FedProx | A | 0.8047 | 0.9176 | 0.6694 | 0.7741 | 0.9188 |
| FedProx | B | 0.8014 | 0.9104 | 0.6684 | 0.7708 | 0.9146 |
| FedProx | C | 0.7528 | 0.9393 | 0.5403 | 0.6860 | 0.9093 |
| Full (FedAlert + Consensus) | A | 0.7697 | 0.9478 | 0.5707 | 0.7124 | 0.9201 |
| Full (FedAlert + Consensus) | B | 0.7864 | 0.9359 | 0.6147 | 0.7420 | 0.9197 |
| Full (FedAlert + Consensus) | C | 0.8035 | 0.8871 | 0.6954 | 0.7796 | 0.9052 |

## Best Performing Models

- **Scenario A**: Consensus Only (F1-Score: 0.8468)
- **Scenario B**: FedAvg (F1-Score: 0.8331)
- **Scenario C**: Consensus Only (F1-Score: 0.8306)