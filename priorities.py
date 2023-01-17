import json5 as json
from pymongo import MongoClient

client = MongoClient()
db = client["JiraRepos"]

with open("./jira_data_sources.json") as f:
    jira_data_sources = json.load(f)


ALL_JIRAS = [jira_name for jira_name in jira_data_sources.keys()]


def priority_fields(jira: str) -> list[str]:
    fields = {
        "Apache": [
            "customfield_12311037",
            "customfield_12311032",
            "customfield_12311027",
            "customfield_12311022",
            "customfield_10023",
        ],
        "Hyperledger": [],
        "IntelDAOS": [],
        "JFrog": ["customfield_12903"],
        "Jira": ["customfield_20234", "customfield_10600", "customfield_10290"],
        "JiraEcosystem": ["customfield_19505"],
        "MariaDB": [],
        "Mindville": [],
        "Mojang": ["customfield_12200"],
        "MongoDB": [],
        "Qt": [],
        "RedHat": [
            "customfield_12317256",
            "customfield_12314440",
            "customfield_12313442",
            "customfield_12312941",
            "customfield_12312940",
            "customfield_12312340",
        ],
        "Sakai": [],
        "SecondLife": [],
        "Sonatype": ["customfield_13603"],
        "Spring": [],
    }

    custom_fields = fields[jira]
    custom_fields = map(lambda x: f"fields.{x}.value", custom_fields)
    custom_fields = list(custom_fields)

    return ["fields.priority.name"] + custom_fields


def extract_field_priorities(jira: str, field: str):
    return db[jira].distinct(field)


for jira in ["RedHat"]:
    priorities = set()

    fields = priority_fields(jira)
    for field in fields:
        field_priorities = extract_field_priorities(jira, field)
        priorities.update(field_priorities)

    print(f"{jira}: {priorities}")
