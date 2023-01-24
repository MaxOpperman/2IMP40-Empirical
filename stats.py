import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_colwidth", None)  # We want to see all data
from statistics import mean, median

client = MongoClient()
db = client["JiraRepos"]

ALL_JIRAS = [
    "Apache",
    "Hyperledger",
    "IntelDAOS",
    "JFrog",
    "Jira",
    "JiraEcosystem",
    "MariaDB",
    "Mindville",
    "Mojang",
    "MongoDB",
    "Qt",
    "RedHat",
    "Sakai",
    "SecondLife",
    "Sonatype",
    "Spring",
]

VERY_LOW = "Very Low"
LOW = "Low"
NORMAL = "Normal"
HIGH = "High"
VERY_HIGH = "Very High"

STD_PRIORITIES = [VERY_LOW, LOW, NORMAL, HIGH, VERY_HIGH]


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
    custom_fields = map(lambda x: f"$fields.{x}.value", custom_fields)
    custom_fields = list(custom_fields)

    # Two times the default prority field as $isNull requires at least 2 params
    return ["$fields.priority.name", "$fields.priority.name"] + custom_fields


def prior_to_repo_prior(jira: str, priority: str) -> list[str]:
    priority_map = {
        "Apache": {
            VERY_LOW: ["Low", "Trivial", "P5", "P4", "Not a Priority", "fp5"],
            LOW: ["P3", "Normal", "Minor", "fp4"],
            NORMAL: ["P2", "High", "Major", "fp3"],
            HIGH: ["P1", "Urgent", "Critical", "fp2"],
            VERY_HIGH: ["Blocker", "P0", "fp1"],
        },
        "Hyperledger": {
            VERY_LOW: ["Lowest"],
            LOW: ["Low"],
            NORMAL: ["Medium"],
            HIGH: ["High"],
            VERY_HIGH: ["Highest"],
        },
        "IntelDAOS": {
            VERY_LOW: ["TBD", "Lowest", "P5-Trivial"],
            LOW: ["P4-Low", "Low"],
            NORMAL: ["P3-Medium"],
            HIGH: ["P2-High"],
            VERY_HIGH: ["P1-Urgent"],
        },
        "JFrog": {
            VERY_LOW: ["6 - Trivial"],
            LOW: ["5 - Minor"],
            NORMAL: ["4 - Normal"],
            HIGH: ["3 - High", "2 - Critical"],
            VERY_HIGH: ["1 - Blocker"],
        },
        "Jira": {
            VERY_LOW: ["5. Lowest", "5"],
            LOW: ["4. Low", "4", "Low"],
            NORMAL: ["3. Medium", "Medium", "3"],
            HIGH: ["2. High", "2", "High"],
            VERY_HIGH: ["Highest", "1. Highest", "1"],
        },
        "JiraEcosystem": {
            VERY_LOW: ["Trivial"],
            LOW: ["Minor"],
            NORMAL: ["Major"],
            HIGH: ["Critical"],
            VERY_HIGH: ["Blocker"],
        },
        "MariaDB": {
            VERY_LOW: ["Trivial"],
            LOW: ["Minor"],
            NORMAL: ["Major"],
            HIGH: ["Critical"],
            VERY_HIGH: ["Blocker"],
        },
        "Mindville": {
            VERY_LOW: ["Lowest"],
            LOW: ["Level 4", "Low"],
            NORMAL: ["Level 3", "Medium"],
            HIGH: ["Level 2", "High"],
            VERY_HIGH: ["Level 1", "Highest"],
        },
        "Mojang": {
            VERY_LOW: ["Low"],
            LOW: ["Normal"],
            NORMAL: ["Important"],
            HIGH: ["Critical", "Very Important"],
            VERY_HIGH: ["Blocker"],
        },
        "MongoDB": {
            VERY_LOW: ["Trivial - P5", "Unknown"],
            LOW: ["Minor - P4"],
            NORMAL: ["Major - P3"],
            HIGH: ["Critical - P2"],
            VERY_HIGH: ["Blocker - P1"],
        },
        "Qt": {
            VERY_LOW: ["Not Evaluated", "P5: Not important"],
            LOW: ["P4: Low"],
            NORMAL: ["P2: Important", "P3: Somewhat important"],
            HIGH: ["P1: Critical"],
            VERY_HIGH: ["P0: Blocker"],
        },
        "RedHat": {
            VERY_LOW: ["Trivial", "Undefined", "Unprioritized"],
            LOW: ["Low", "Minor", "Optional"],
            NORMAL: ["High", "Major", "Should Have", "Normal", "Medium"],
            HIGH: ["Urgent", "Must Have", "Critical"],
            VERY_HIGH: ["Blocker"],
        },
        "Sakai": {
            VERY_LOW: [],
            LOW: ["Minor"],
            NORMAL: ["Major"],
            HIGH: ["Critical"],
            VERY_HIGH: ["Blocker"],
        },
        "SecondLife": {
            VERY_LOW: ["Trivial", "Unset"],
            LOW: ["Minor"],
            NORMAL: ["Major"],
            HIGH: ["Severe"],
            VERY_HIGH: ["Showstopper"],
        },
        "Sonatype": {
            VERY_LOW: ["Trivial"],
            LOW: ["Minor"],
            NORMAL: ["Major", "Medium", "3 - Normal"],
            HIGH: ["Critical", "Complex Fast-Track"],
            VERY_HIGH: ["Blocker"],
        },
        "Spring": {
            VERY_LOW: ["Trivial"],
            LOW: ["Minor"],
            NORMAL: ["Major"],
            HIGH: ["Critical"],
            VERY_HIGH: ["Blocker"],
        },
    }

    return priority_map[jira][priority]


def generate_conditions(jira: str, priorities: list[str] = STD_PRIORITIES):
    priority = priorities[0]
    if len(priorities) == 1:
        return {
            "$cond": {
                "if": {
                    "$in": [
                        "$full_priority",
                        prior_to_repo_prior(jira, priority),
                    ]
                },
                "then": priority,
                "else": "NO PRIORITY SHOULD NOT HAPPEN",
            }
        }

    return {
        "$cond": {
            "if": {
                "$in": [
                    "$full_priority",
                    prior_to_repo_prior(jira, priority),
                ]
            },
            "then": priority,
            "else": generate_conditions(jira, priorities[1:]),
        }
    }


def normalize(jira: str):
    return [
        {"$addFields": {"full_priority": {"$ifNull": priority_fields(jira)}}},
        {"$match": {"full_priority": {"$ne": None}}},
        {"$addFields": {"std_priority": generate_conditions(jira)}},
    ]


def compute_rq_1(jiras: list[str] = ALL_JIRAS):
    def extract_mean_date_diff(jira: str):
        return list(
            db[jira].aggregate(
                normalize(jira) +
                [
                    {
                        "$match": {
                            "fields.resolutiondate": {"$exists": True, "$ne": None}
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
                                },
                                '?<?', {
                                    "$cond": {
                                        "if": {
                                            "$in": [
                                                "$std_priority",
                                                [HIGH, VERY_HIGH],
                                            ]
                                        },
                                        "then": HIGH,
                                        "else": LOW,
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
                        },
                        'dateDifferences': {
                            '$push': {
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
                        }
                    }
                }, {
                    '$set': {
                        'dateDifferences': {
                            '$sortArray': {
                                'input': '$dateDifferences',
                                'sortBy': 1
                            }
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
                        "week": {
                            "$toInt": {
                                "$arrayElemAt": [{"$split": ["$_id", "?<?"]}, 1]
                            }
                        },
                        "priorityLevel": {"$last": {"$split": ["$_id", "?<?"]}},
                        "medianDateDiff": {
                            "$arrayElemAt": [
                                "$dateDifferences",
                                {
                                    "$floor": {
                                        "$divide": [
                                            {"$size": "$dateDifferences"},
                                            2,
                                        ]
                                    }
                                },
                            ]
                        },
                        "avgSecondsDiff": 1,
                        "count": 1,
                    }
                },
                {"$sort": {"year": -1, "week": -1, "priorityLevel": 1}},
                ]
            )
        )

    def create_df(jira: str):
        date_diff_records = extract_mean_date_diff(jira)

        date_df = pd.DataFrame(date_diff_records)

        # remove the values from the list that do not have both high and low priority lists
        duplicates_df = date_df.loc[
            date_df.duplicated(subset=["year", "week"], keep=False)
        ].reset_index(drop=True)

        factors = []
        # factors = [avgLow, avgHigh, countLow, countHigh, factor (high/low)]
        for index, row in duplicates_df.iterrows():
            if row["priorityLevel"] == HIGH:
                low_row = duplicates_df.loc[(duplicates_df['year'] == row['year']) & (
                        duplicates_df['week'] == row['week']) & (duplicates_df['priorityLevel'] == LOW)]
                factors.append(
                    [
                        f"{row['year']}/{row['week']}",
                        low_row.iloc[0]["avgSecondsDiff"],
                        row["avgSecondsDiff"],
                        low_row.iloc[0]["medianDateDiff"],
                        row["medianDateDiff"],
                        low_row.iloc[0]["count"],
                        row["count"],
                        (row["avgSecondsDiff"] / low_row.iloc[0]["avgSecondsDiff"]),
                        (row["medianDateDiff"] / low_row.iloc[0]["medianDateDiff"]),
                    ]
                )

        factor_df = pd.DataFrame(
            factors,
            columns=["creationDate", "avgLow", "avgHigh", "medianLow", "medianHigh", "countLow", "countHigh",
                     "averageFactor", "medianFactor"],
        )
        print(
            f'Count Low: {factor_df["countLow"].sum()}, High: {factor_df["countHigh"].sum()}.\n'
            f'Averages factor: avg {factor_df["averageFactor"].mean()}, median {factor_df["averageFactor"].median()}.\n'
            f'Medians factor: avg {factor_df["medianFactor"].mean()}, median {factor_df["medianFactor"].median()}'
        )

        factor_df = factor_df.sort_values(by=['medianFactor'])

        # plt.bar(factor_df['creationDate'], factor_df['averageFactor'])
        plt.bar(factor_df['creationDate'], factor_df['medianFactor'])
        plt.show()

        with open("factors.csv", "w+") as f:
            factor_df.to_csv(f, header=True)

        return date_df

    print("\nComputing stats for RQ1\n")

    for jira in jiras[2:]:
        print(f"Processing {jira}...")
        df = create_df(jira)
        print(df, "\n")


def compute_rq_2(jiras: list[str] = ALL_JIRAS):
    def filter_issue_links():
        bad_issue_types = [
            # Clone
            "Cloners",
            "Cloners (old)",
            # Duplicate
            "Duplicate",
            # Split
            "Issue split",
            "Split",
            "Work Breakdown",
        ]

        return [
            {
                "$set": {
                    "issuelinks": {
                        "$map": {
                            "input": "$fields.issuelinks",
                            "as": "issue",
                            "in": "$$issue.type.name",
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "std_priority": 1,
                    "issuelinks": {
                        "$filter": {
                            "input": "$issuelinks",
                            "as": "link",
                            "cond": {"$not": {"$in": ["$$link", bad_issue_types]}},
                        }
                    },
                }
            },
        ]

    def extract_mean_and_count(jira: str):
        return list(
            db[jira].aggregate(
                normalize(jira)
                + filter_issue_links()
                + [
                    {
                        "$group": {
                            "_id": "$std_priority",
                            "issuelink_sizes": {"$push": {"$size": "$issuelinks"}},
                        }
                    },
                ]
            )
        )

    def create_df(jira: str):
        df = pd.DataFrame(
            0,
            index=STD_PRIORITIES,
            columns=["Median Links"],
        )

        records = extract_mean_and_count(jira)

        for record in records:
            priority_name = record["_id"]
            df.loc[priority_name, "Median Links"] = median(record["issuelink_sizes"])

        return df

    print("\nComputing stats for RQ2\n")

    df_final = pd.DataFrame(0, index=STD_PRIORITIES, columns=ALL_JIRAS + ["Aggregate"])

    for jira in jiras:
        print(f"Processing {jira}...")
        df = create_df(jira)

        for priority in STD_PRIORITIES:
            df_final.loc[priority, jira] = df.loc[priority, "Median Links"]

    print(df_final)


compute_rq_1()
# compute_rq_2()
