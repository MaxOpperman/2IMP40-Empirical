import json5 as json


def is_custom_priority_field(field):
    return "prio" in field["name"].lower() and field["custom"]


with open("./jira_field_information.json") as f:
    field_information = json.load(f)

for jira in field_information:
    custom_fields = field_information[jira]
    custom_fields = filter(is_custom_priority_field, custom_fields)
    custom_fields = map(lambda x: x["id"], custom_fields)
    custom_fields = sorted(custom_fields, reverse=True)
    print(f"{jira}: {custom_fields}")
