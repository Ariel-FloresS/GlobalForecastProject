from dataclasses import dataclass


@dataclass(frozen=True)
class DatabricksConnectConfig:

    host: str
    token: str
    cluster_id: str

