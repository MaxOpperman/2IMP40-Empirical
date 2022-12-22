from datetime import timedelta

import json5 as json
import numpy as np
import pandas as pd
from pymongo import MongoClient

pd.set_option("display.max_colwidth", None)  # We want to see all data
from statistics import mean, median

client = MongoClient()
db = client["JiraRepos"]

# Load the Jira Data Sources JSON
with open("./jira_data_sources.json") as f:
    jira_data_sources = json.load(f)

# Load the Jira Issue Types Information (Downloaded using the DataDownload script)
with open("./jira_issuetype_information.json") as f:
    jira_issuetype_information = json.load(f)

# Load the Jira Issue Link Types Information (Downloaded using the DataDownload script)
with open("./jira_issuelinktype_information.json") as f:
    jira_issuelinktype_information = json.load(f)

# Load the Jira Thematic Analysis JSON
# with open('./jira_issuetype_thematic_analysis.json') as f:
#     issuetype_themes_codes = json.load(f)

ALL_JIRAS = [jira_name for jira_name in jira_data_sources.keys()]


def create_df(jira):
    def extract_mean_and_count():
        return list(
            db[jira].aggregate(
                [
                    {
                        "$match": {
                            "fields.issuelinks": {"$type": "array"},
                        }
                    },
                    {
                        "$group": {
                            "_id": "$fields.priority",
                            "avg": {
                                "$avg": {"$size": "$fields.issuelinks"},
                            },
                            "count": {"$count": {}},
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "name": "$_id.name",
                            "avg": 1,
                            "count": 1,
                        }
                    },
                ]
            )
        )

    def extract_mean_date_diff():
        return list(
            db[jira].aggregate(
                [
                    {
                        '$match': {
                            'fields.resolutiondate': {
                                '$exists': True,
                                '$ne': None
                            }
                        }
                    }, {
                        '$group': {
                            '_id': {
                                '$concat': [
                                    {
                                        '$toString': {
                                            '$year': {
                                                '$toDate': '$fields.created'
                                            }
                                        }
                                    }, '?<?', {
                                        '$toString': {
                                            '$week': {
                                                '$toDate': '$fields.created'
                                            }
                                        }
                                    }, '?<?', {
                                        '$toString': {
                                            '$in': [
                                                '$fields.priority.name', [
                                                    'Blocker', 'Complex Fast-Track', 'Critical', 'High', 'Highest', 'Major'
                                                ]
                                            ]
                                        }
                                    }
                                ]
                            },
                            'avgSecondsDiff': {
                                '$avg': {
                                    '$dateDiff': {
                                        'startDate': {
                                            '$toDate': '$fields.created'
                                        },
                                        'endDate': {
                                            '$toDate': '$fields.resolutiondate'
                                        },
                                        'unit': 'second'
                                    }
                                }
                            },
                            'count': {
                                '$count': {}
                            }
                        }
                    }, {
                        '$project': {
                            'year': {
                                '$toInt': {
                                    '$first': {
                                        '$split': [
                                            '$_id', '?<?'
                                        ]
                                    }
                                }
                            },
                            'week': {
                                '$toInt': {
                                    '$arrayElemAt': [
                                        {
                                            '$split': [
                                                '$_id', '?<?'
                                            ]
                                        }, 1
                                    ]
                                }
                            },
                            'priorityLevel': {
                                '$last': {
                                    '$split': [
                                        '$_id', '?<?'
                                    ]
                                }
                            },
                            'avgSecondsDiff': 1,
                            'count': 1
                        }
                    }, {
                        '$sort': {
                            'year': -1,
                            'week': -1,
                            'priorityLevel': 1
                        }
                    }
                ]
            )
        )

    def get_priority_name(record, default="NULL"):
        return record.get("name", default)

    def get_priorities(records):
        return [get_priority_name(record) for record in records]

    records = extract_mean_and_count()

    df = pd.DataFrame(
        np.nan,
        index=get_priorities(records),
        columns=["Mean Link Count", "PriorCount", "Mean Date Diff", "DateCount"],
    )

    for record in records:
        priority_name = get_priority_name(record)
        df.loc[priority_name, "Mean Link Count"] = record["avg"]
        df.loc[priority_name, "PriorCount"] = int(record["count"])

    date_diff_records = extract_mean_date_diff()

    date_df = pd.DataFrame(date_diff_records)

    # remove the values from the list that do not have both high and low priority lists
    duplicates_df = date_df.loc[date_df.duplicated(subset=['year', 'week'], keep=False)].reset_index(drop=True)

    factors = []
    # factors = [avgLow, avgHigh, countLow, countHigh, factor (high/low)]
    for index in range(0, len(duplicates_df)):
        if duplicates_df.iloc[index]["priorityLevel"] == "false":
            low_prio = duplicates_df.iloc[index]
            high_prio = duplicates_df.iloc[index+1]
            factors.append([
                low_prio['avgSecondsDiff'],
                high_prio['avgSecondsDiff'],
                low_prio['count'],
                high_prio['count'],
                (high_prio['avgSecondsDiff'] / low_prio['avgSecondsDiff'])
            ])

    factor_df = pd.DataFrame(factors, columns=["avgLow", "avgHigh", "countLow", "countHigh", "Factor(high/low)"])

    print(factor_df)
    with open('factors.csv', 'w+') as f:
        factor_df.to_csv(f, header=True)
    print(f'Count Low: {factor_df["countLow"].sum()}, High: {factor_df["countHigh"].sum()}. Average factor: {factor_df["Factor(high/low)"].mean()}.')

    return df


def compute_stats_1(jiras=ALL_JIRAS):
    for jira in jiras:
        print(f"Processing {jira}...")
        df = create_df(jira)
        print(df, "\n")


compute_stats_1()


###
### Old Stuff
###
def old():
    # These are the global dataframes that we will perform our analysis on.
    df_jiras = pd.DataFrame(
        np.nan,
        columns=[
            "Born",
            "Issues",
            "DIT",
            "UIT",
            "Links",
            "DLT",
            "ULT",
            "Changes",
            "Ch/I",
            "UP",
            "Comments",
            "Co/I",
        ],
        index=ALL_JIRAS + ["Sum", "Median", "Std Dev"],
    )

    def populate_df_jiras(df_jiras, jiras=ALL_JIRAS):
        def extract_number_of_issues(jira_name):
            # Query for the count of all issues
            num_issues = db[jira_name].count_documents({})
            # Return value
            return num_issues

        def extract_number_of_documented_issuetypes(jira_name):
            # Extract the number of documented issue types from the downloaded issuetype_information JSON downloaded earlier, and return
            return len(jira_issuetype_information[jira_name])

        def extract_number_of_used_issuetypes(jira_name):
            # Query for unique set of issuetypes in the final state of the issue
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # We only need the issuetype name for the final state evaluation
                        {
                            "$project": {
                                "_id": 0,
                                "issuetype_name": "$fields.issuetype.name",
                            }
                        },
                        # Create a unique set of these names
                        {
                            "$group": {
                                "_id": None,
                                "issuetype_names": {"$addToSet": "$issuetype_name"},
                            }
                        },
                    ]
                )
            )
            # Extract the query
            unique_issuetypes_final = (
                set(query_result[0]["issuetype_names"]) if query_result else set()
            )
            # Query for unique set of issuetypes in the issue history
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # Unwind the histories and items to work with individual change items
                        {"$unwind": "$changelog.histories"},
                        {"$unwind": "$changelog.histories.items"},
                        # We only want the changes to the 'issuetype' field
                        {"$match": {"changelog.histories.items.field": "issuetype"}},
                        # Select and rename the nested 'fromString' attribute. We only care what the issueType was BEFORE changing.
                        # We have the subsequent 'toString' values in the next change 'fromString' or the final state extracted above.
                        {
                            "$project": {
                                "_id": 0,
                                "issuetype_name": "$changelog.histories.items.fromString",
                            }
                        },
                        # Create a unique set of these names
                        {
                            "$group": {
                                "_id": None,
                                "issuetype_names": {"$addToSet": "$issuetype_name"},
                            }
                        },
                    ]
                )
            )
            # Extract the query
            unique_issuetypes_history = (
                set(query_result[0]["issuetype_names"]) if query_result else 0
            )
            # Union the two sets together, and count the items, and return
            return len(set.union(unique_issuetypes_final, unique_issuetypes_history))

        def extract_number_of_issuelinks(jira_name):
            # Extract the issuelinks
            issuelinks_result = list(
                db[jira_name].aggregate(
                    [
                        # Limit to issues with issuelinks
                        {"$match": {"fields.issuelinks": {"$exists": True, "$ne": []}}},
                        # Limit the object data to just the issuelink ids, and rename/condense into a single field
                        {
                            "$project": {
                                "_id": 0,
                                "issuelink_ids_issue": "$fields.issuelinks.id",
                            }
                        },
                        # Create a new "row" for each issue link, since issues can have multiple issuelinks each
                        {"$unwind": "$issuelink_ids_issue"},
                        # Create a unique set of issuelink ids. Issuelinks link multiple issues together, but we only want to count this link once.
                        {
                            "$group": {
                                "_id": None,
                                "issuelink_unique_ids": {
                                    "$addToSet": "$issuelink_ids_issue"
                                },
                            }
                        },
                    ]
                )
            )
            num_issuelinks = (
                len(set(issuelinks_result[0]["issuelink_unique_ids"]))
                if issuelinks_result
                else 0
            )
            # Extract the subtasks
            subtasks_result = list(
                db[jira_name].aggregate(
                    [
                        # Limit to issues with subtasks
                        {"$match": {"fields.subtasks": {"$exists": True, "$ne": []}}},
                        # Limit the object data to just the size of the subtask arrays.
                        {
                            "$project": {
                                "_id": 0,
                                "num_issue_subtasks": {"$size": "$fields.subtasks"},
                            }
                        },
                        # Count the subtask arrays across the entire jira dataset
                        {
                            "$group": {
                                "_id": None,
                                "num_subtasks": {"$sum": "$num_issue_subtasks"},
                            }
                        },
                    ]
                )
            )
            num_subtasks = subtasks_result[0]["num_subtasks"] if subtasks_result else 0
            # Extract the epic links
            epiclinkfield_dict = {
                "Apache": "customfield_12311120",
                "Hyperledger": "customfield_10006",
                "IntelDAOS": "customfield_10092",
                "JFrog": "customfield_10806",
                "Jira": "customfield_12931",
                "JiraEcosystem": "customfield_12180",
                "MariaDB": "customfield_10600",
                "Mindville": "customfield_10000",
                "Mojang": "customfield_11602",
                "MongoDB": "customfield_10857",
                "Qt": "customfield_10400",
                "RedHat": "customfield_12311140",
                "Sakai": "customfield_10772",
                "SecondLife": "customfield_10871",
                "Sonatype": "customfield_11500",
                "Spring": "customfield_10680",
            }
            epiclinks_result = list(
                db[jira_name].aggregate(
                    [
                        # Rename the field since every Jira uses a different customfield name
                        {
                            "$project": {
                                "epiclink_field": f"$fields.{epiclinkfield_dict[jira_name]}"
                            }
                        },
                        # Limit to issues with epiclink fields
                        {"$match": {"epiclink_field": {"$exists": True, "$ne": None}}},
                        # Count the number of records in the aggregation
                        {"$count": "num_epiclinks"},
                    ]
                )
            )
            num_epiclinks = (
                epiclinks_result[0]["num_epiclinks"] if epiclinks_result else 0
            )  # Some repos have no epiclinks, so we need to catch this
            # Total the number of issuelinks by summing the three values above, and return
            return sum([num_issuelinks, num_subtasks, num_epiclinks])

        def extract_number_of_documented_issuelinktypes(jira_name):
            # Extract the number of documented issue link types from the downloaded issuelinktype_information JSON downloaded earlier, and return
            return (
                len(jira_issuelinktype_information[jira_name])
                if jira_name in jira_issuelinktype_information
                else 0
            )

        def extract_number_of_used_issuelinktypes(jira_name):
            # Query for unique set of issuelinktypes in the final state of the issue
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # Unwind the issuelinks into individual records
                        {"$unwind": "$fields.issuelinks"},
                        # Select and rename the issuelink type name to prepare for the group operator
                        {
                            "$project": {
                                "_id": 0,
                                "issuelinktype_name": "$fields.issuelinks.type.name",
                            }
                        },
                        # Create a unique set of the issuelink type names
                        {
                            "$group": {
                                "_id": None,
                                "issuelinktype_names": {
                                    "$addToSet": "$issuelinktype_name"
                                },
                            }
                        },
                    ]
                )
            )
            # Extract the query, and return value
            return (
                len(set(query_result[0]["issuelinktype_names"])) if query_result else 0
            )

        def extract_born(jira_name):
            # Get the first N issues in each repo to check for the initial "birth" of the repo
            created_dates = [
                issue["fields"]["created"]
                for issue in db[jira_name].aggregate(
                    [
                        # We only need the created field
                        {"$project": {"_id": 0, "fields.created": 1}},
                        # Sort the items by created date (ascending) to get the earliest dates first
                        {"$sort": {"fields.created": 1}},
                        # We only technically need the first item, but practically there are issues that need to be manually reviewed below
                        {"$limit": 500},
                    ]
                )
            ]
            # Manual analaysis of the created dates revealed a number of broken or testing issues that should be ignored
            if jira_name == "Apache":
                created_dates = created_dates[289:]
            elif jira_name == "Jira":
                created_dates = created_dates[1:]
            elif jira_name == "IntelDAOS":
                created_dates = created_dates[1:]
            elif jira_name == "Qt":
                created_dates = created_dates[7:]
            # Return value
            return created_dates[0][:4]

        def extract_number_of_changes(jira_name):
            # Query for the number of changes
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # We only need one attribute of the change to count it
                        {"$project": {"_id": 0, "changelog.histories.items.field": 1}},
                        # Unwind the histories and items arrays into single elements so we can count them
                        {"$unwind": "$changelog.histories"},
                        {"$unwind": "$changelog.histories.items"},
                        # Count number of elements in our aggregation, which is now the number of items
                        {"$count": "num_changes"},
                    ]
                )
            )
            # Extract the query result and return
            return query_result[0]["num_changes"] if query_result else 0

        def extract_number_of_unique_projects(jira_name):
            # Query for a unique set of project ids in the final state of the issue
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # Limit to just the final project name on each issue
                        {
                            "$project": {
                                "_id": 0,
                                "project_name": "$fields.project.name",
                            }
                        },
                        # Create a unique set of project names across the entire Jira
                        {
                            "$group": {
                                "_id": None,
                                "project_names": {"$addToSet": "$project_name"},
                            }
                        },
                    ]
                )
            )
            # Extract the query result
            unique_projects_final = (
                set(query_result[0]["project_names"]) if query_result else set()
            )
            # Query for a unique set of project ids in the issue history
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # Unwind the histories and items to work with individual change items
                        {"$unwind": "$changelog.histories"},
                        {"$unwind": "$changelog.histories.items"},
                        # Select only changes where the project field was changed
                        {"$match": {"changelog.histories.items.field": "project"}},
                        # Rename the nested 'fromString' field containing the previously slected project
                        {
                            "$project": {
                                "_id": 0,
                                "project_name": "$changelog.histories.items.fromString",
                            }
                        },
                        # Create a unique set of these project names
                        {
                            "$group": {
                                "_id": None,
                                "project_names": {"$addToSet": "$project_name"},
                            }
                        },
                    ]
                )
            )
            # Extract the query result
            unique_projects_history = (
                set(query_result[0]["project_names"]) if query_result else set()
            )
            # Union the two sets together, count the items, and return
            return len(set.union(unique_projects_final, unique_projects_history))

        def extract_number_of_comments(jira_name):
            # Query for the number of changes
            query_result = list(
                db[jira_name].aggregate(
                    [
                        # Get issues with comments
                        {"$match": {"fields.comments": {"$ne": None}}},
                        # We only need one attribute of the change to count it
                        {
                            "$project": {
                                "_id": 0,
                                "num_comments_per_issue": {"$size": "$fields.comments"},
                            }
                        },
                        # Group the sizes so we can sum all to a single value for the repo
                        {
                            "$group": {
                                "_id": None,
                                "num_comments": {"$sum": "$num_comments_per_issue"},
                            }
                        },
                    ]
                )
            )
            # Extract the query result and return
            return query_result[0]["num_comments"] if query_result else 0

        print("This script takes ~90 minutes when executed across all Jiras.")

        # Populate the table with the answers to our questions
        for jira_name in jiras:
            print(f"\tWorking on Jira: {jira_name} ...")

            ## Issues and their Types ##

            # Attribute: Issues (number of issues)
            df_jiras.loc[jira_name, "Issues"] = extract_number_of_issues(jira_name)
            # Attribute: DIT (documented issue types)
            df_jiras.loc[jira_name, "DIT"] = extract_number_of_documented_issuetypes(
                jira_name
            )
            # Attribute: UIT (used issue types)
            df_jiras.loc[jira_name, "UIT"] = extract_number_of_used_issuetypes(
                jira_name
            )

            ## Issue Links and their Types ##

            # Attribute: Links (number of links)
            df_jiras.loc[jira_name, "Links"] = extract_number_of_issuelinks(jira_name)
            # Attribute: DLT (documented link types)
            df_jiras.loc[
                jira_name, "DLT"
            ] = extract_number_of_documented_issuelinktypes(jira_name)
            # Attribute: ULD (used link types)
            df_jiras.loc[jira_name, "ULT"] = extract_number_of_used_issuelinktypes(
                jira_name
            )

            ## General Information ##

            # Attribute: Born (first issue added)
            df_jiras.loc[jira_name, "Born"] = extract_born(jira_name)
            # Attribute: Changes (number of changes)
            df_jiras.loc[jira_name, "Changes"] = extract_number_of_changes(jira_name)
            # Attribute: Ch/I (number of changes per issue)
            df_jiras.loc[jira_name, "Ch/I"] = round(
                df_jiras.loc[jira_name, "Changes"] / df_jiras.loc[jira_name, "Issues"]
            )
            # Attribute: UP (unique projects)
            df_jiras.loc[jira_name, "UP"] = extract_number_of_unique_projects(jira_name)
            # Attribute: Comments (number of comments)
            df_jiras.loc[jira_name, "Comments"] = extract_number_of_comments(jira_name)
            # Attribute: Co/I (number of comments per issue)
            df_jiras.loc[jira_name, "Co/I"] = round(
                df_jiras.loc[jira_name, "Comments"] / df_jiras.loc[jira_name, "Issues"]
            )

        print("Complete")
        return df_jiras

    df_jiras = populate_df_jiras(
        df_jiras,
        ## Test to see if the script works (database created, data inside, etc.) ##
        # jiras=['Hyperledger'],
        ## To test the script in less than 90 minutes, uncomment the following line and see the result of a few select Jira repos ##
        # jiras=['Hyperledger', 'IntelDAOS', 'JFrog', 'Sakai', 'SecondLife', 'Sonatype', 'Spring'],
    )

    def display_df_jiras(df_jiras):
        # Complete final summative rows
        for header in df_jiras.columns:
            if header in ["Born"]:
                continue
            df_jiras.loc["Sum", header] = sum(df_jiras[header][: len(ALL_JIRAS)])
            df_jiras.loc["Median", header] = median(df_jiras[header][: len(ALL_JIRAS)])
            df_jiras.loc["Std Dev", header] = np.std(df_jiras[header][: len(ALL_JIRAS)])

        # Columns to comma-separate
        # comma_separated_columns = {col_name: '{:,.0f}' for col_name in ['Issues', 'Links', 'Changes', 'Comments']}

        # Display the data
        print(
            df_jiras
            # .style
            # .set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
            # .format(
            #     comma_separated_columns,
            #     precision=0
            # )
        )

    display_df_jiras(df_jiras)
