"""Historical CFP and BCS bowl game dataset (36 games, 2006-2025) and filter helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from models import Game

HISTORICAL_GAMES: list[dict[str, Any]] = [
    # 2024-25 CFP
    {
        "id": "2025-cotton-bowl",
        "game": "Cotton Bowl - CFP Quarterfinal",
        "year": 2025,
        "date": "2024-12-31",
        "teams": {"winner": "Miami", "loser": "Ohio State"},
        "seeds": {"winner": 10, "loser": 2},
        "score": {"winner": 24, "loser": 14},
        "venue": "AT&T Stadium",
        "narrative": (
            "In a stunning upset, 10-seed Miami dominated 2-seed Ohio State from start to "
            "finish. The Hurricanes built a 14-0 halftime lead and never looked back, becoming "
            "the first double-digit seed to reach a CFP semifinal. Ohio State's vaunted offense "
            "was held to just 14 points as Miami's defense forced three turnovers. The Buckeyes "
            "entered as heavy favorites but were outplayed in every phase."
        ),
        "key_stats": {"upset": True, "margin": 10, "turnovers_forced": 3},
        "tags": ["upset", "defensive_battle", "cfp_quarterfinal", "double_digit_seed"],
    },
    {
        "id": "2025-first-round-alabama",
        "game": "CFP First Round",
        "year": 2025,
        "date": "2024-12-19",
        "teams": {"winner": "Alabama", "loser": "Oklahoma"},
        "seeds": {"winner": 9, "loser": 8},
        "score": {"winner": 34, "loser": 24},
        "venue": "Oklahoma Memorial Stadium",
        "narrative": (
            "Alabama overcame a 17-point first-half deficit to stun Oklahoma on the road. The "
            "Crimson Tide became the first team in the 12-team playoff era to win a road game. "
            "Alabama's second-half surge was powered by their rushing attack, which wore down "
            "Oklahoma's defense. The comeback marked one of the largest deficits overcome in "
            "CFP history."
        ),
        "key_stats": {"comeback": 17, "margin": 10, "road_win": True},
        "tags": ["comeback", "road_win", "cfp_first_round", "historic"],
    },
    # 2023-24 CFP
    {
        "id": "2024-cfp-championship",
        "game": "CFP National Championship",
        "year": 2024,
        "date": "2024-01-08",
        "teams": {"winner": "Michigan", "loser": "Washington"},
        "seeds": {"winner": 1, "loser": 2},
        "score": {"winner": 34, "loser": 13},
        "venue": "NRG Stadium",
        "narrative": (
            "Michigan dominated Washington to claim its first national title since 1997. The "
            "Wolverines' suffocating defense held Washington's high-powered offense to just 13 "
            "points, while Blake Corum and the rushing attack controlled the clock. Michigan "
            "became the first team to go 15-0 in the CFP era. Coach Jim Harbaugh delivered on "
            "his promise to bring a championship back to Ann Arbor."
        ),
        "key_stats": {"margin": 21, "undefeated": True, "rushing_yards": 303},
        "tags": ["championship", "dominant", "undefeated_champion", "defensive_masterpiece"],
    },
    {
        "id": "2024-rose-bowl-semi",
        "game": "Rose Bowl - CFP Semifinal",
        "year": 2024,
        "date": "2024-01-01",
        "teams": {"winner": "Michigan", "loser": "Alabama"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 27, "loser": 20},
        "venue": "Rose Bowl",
        "narrative": (
            "Michigan held off a furious Alabama comeback in overtime to advance to the "
            "national championship. Alabama erased a 13-point fourth-quarter deficit to force "
            "OT, but Michigan's defense stood tall when it mattered most. The Wolverines' Blake "
            "Corum had a workmanlike performance, and the defense made decisive stops in "
            "overtime."
        ),
        "key_stats": {"overtime": True, "margin": 7, "comeback_survived": 13},
        "tags": ["semifinal", "overtime", "classic", "defensive_stand"],
    },
    {
        "id": "2024-sugar-bowl-semi",
        "game": "Sugar Bowl - CFP Semifinal",
        "year": 2024,
        "date": "2024-01-01",
        "teams": {"winner": "Washington", "loser": "Texas"},
        "seeds": {"winner": 2, "loser": 3},
        "score": {"winner": 37, "loser": 31},
        "venue": "Caesars Superdome",
        "narrative": (
            "Michael Penix Jr. threw for 430 yards and 2 touchdowns as Washington outlasted "
            "Texas in a thriller. The Huskies' air raid offense proved unstoppable, with Rome "
            "Odunze hauling in 10 catches. Texas fought back from multiple deficits but "
            "couldn't stop Penix in the final minutes."
        ),
        "key_stats": {"passing_yards": 430, "margin": 6, "high_scoring": True},
        "tags": ["semifinal", "shootout", "qb_duel", "offensive_showcase"],
    },
    # 2022-23 CFP
    {
        "id": "2023-cfp-championship",
        "game": "CFP National Championship",
        "year": 2023,
        "date": "2023-01-09",
        "teams": {"winner": "Georgia", "loser": "TCU"},
        "seeds": {"winner": 1, "loser": 3},
        "score": {"winner": 65, "loser": 7},
        "venue": "SoFi Stadium",
        "narrative": (
            "Georgia delivered the most dominant championship performance in CFP history, "
            "crushing Cinderella TCU 65-7. The Bulldogs scored on their first seven possessions "
            "and held TCU to just 188 total yards. Stetson Bennett threw for 6 touchdowns as "
            "Georgia became the first repeat champion since Alabama in 2011-12."
        ),
        "key_stats": {"margin": 58, "touchdowns": 9, "largest_cfp_margin": True},
        "tags": ["championship", "blowout", "historic_margin", "repeat_champion"],
    },
    {
        "id": "2023-fiesta-bowl-semi",
        "game": "Fiesta Bowl - CFP Semifinal",
        "year": 2023,
        "date": "2022-12-31",
        "teams": {"winner": "TCU", "loser": "Michigan"},
        "seeds": {"winner": 3, "loser": 2},
        "score": {"winner": 51, "loser": 45},
        "venue": "State Farm Stadium",
        "narrative": (
            "TCU's magical season continued with an epic overtime win over Michigan. The Horned "
            "Frogs trailed by 14 in the fourth quarter before Max Duggan led a furious rally. "
            "The game featured 96 combined points and multiple lead changes in the final "
            "minutes. TCU's improbable run from unranked to the championship game captured the "
            "nation's imagination."
        ),
        "key_stats": {"overtime": True, "combined_points": 96, "comeback": 14},
        "tags": ["semifinal", "overtime", "shootout", "cinderella", "instant_classic"],
    },
    {
        "id": "2023-peach-bowl-semi",
        "game": "Peach Bowl - CFP Semifinal",
        "year": 2023,
        "date": "2022-12-31",
        "teams": {"winner": "Georgia", "loser": "Ohio State"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 42, "loser": 41},
        "venue": "Mercedes-Benz Stadium",
        "narrative": (
            "Georgia survived a Peach Bowl classic, holding off Ohio State 42-41. The Bulldogs "
            "overcame a 14-point deficit in the fourth quarter, with Stetson Bennett "
            "engineering the game-winning drive. CJ Stroud threw for 348 yards in a losing "
            "effort. Georgia's defense made a decisive goal-line stand in the final seconds."
        ),
        "key_stats": {"margin": 1, "combined_points": 83, "goal_line_stand": True},
        "tags": ["semifinal", "classic", "comeback", "goal_line_stand", "thriller"],
    },
    # 2021-22 CFP
    {
        "id": "2022-cfp-championship",
        "game": "CFP National Championship",
        "year": 2022,
        "date": "2022-01-10",
        "teams": {"winner": "Georgia", "loser": "Alabama"},
        "seeds": {"winner": 3, "loser": 1},
        "score": {"winner": 33, "loser": 18},
        "venue": "Lucas Oil Stadium",
        "narrative": (
            "Georgia finally slayed the dragon, defeating Alabama for the national championship "
            "after losing to the Tide in the SEC Championship just weeks earlier. Stetson "
            "Bennett, the former walk-on, threw for 224 yards and 2 touchdowns. The Bulldogs' "
            "defense picked off Bryce Young twice in the fourth quarter to seal the victory and "
            "end a 41-year national title drought."
        ),
        "key_stats": {"margin": 15, "interceptions": 2, "revenge_game": True},
        "tags": ["championship", "revenge", "title_drought_ended", "defensive_dominance"],
    },
    {
        "id": "2022-cotton-bowl-semi",
        "game": "Cotton Bowl - CFP Semifinal",
        "year": 2022,
        "date": "2021-12-31",
        "teams": {"winner": "Alabama", "loser": "Cincinnati"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 27, "loser": 6},
        "venue": "AT&T Stadium",
        "narrative": (
            "Alabama's defense stifled Cincinnati's first-ever CFP appearance. The Bearcats, "
            "the first Group of Five team to make the playoff, couldn't find an answer for Will "
            "Anderson Jr. and the Tide's pass rush. Bryce Young threw for 181 yards and 3 "
            "touchdowns in a workmanlike victory."
        ),
        "key_stats": {"margin": 21, "sacks": 6, "first_g5_team": True},
        "tags": ["semifinal", "defensive_dominance", "g5_underdog", "sack_fest"],
    },
    {
        "id": "2022-orange-bowl-semi",
        "game": "Orange Bowl - CFP Semifinal",
        "year": 2022,
        "date": "2021-12-31",
        "teams": {"winner": "Georgia", "loser": "Michigan"},
        "seeds": {"winner": 3, "loser": 2},
        "score": {"winner": 34, "loser": 11},
        "venue": "Hard Rock Stadium",
        "narrative": (
            "Georgia's vaunted defense overwhelmed Michigan in the Orange Bowl. The Bulldogs "
            "held the Wolverines to just 88 rushing yards after Michigan had averaged over 200 "
            "on the ground all season. Georgia forced two turnovers and sacked JJ McCarthy four "
            "times in a dominant defensive performance."
        ),
        "key_stats": {"margin": 23, "sacks": 4, "rushing_yards_allowed": 88},
        "tags": ["semifinal", "defensive_masterpiece", "dominant"],
    },
    # 2020-21 CFP
    {
        "id": "2021-cfp-championship",
        "game": "CFP National Championship",
        "year": 2021,
        "date": "2021-01-11",
        "teams": {"winner": "Alabama", "loser": "Ohio State"},
        "seeds": {"winner": 1, "loser": 3},
        "score": {"winner": 52, "loser": 24},
        "venue": "Hard Rock Stadium",
        "narrative": (
            "DeVonta Smith put on one of the greatest championship performances ever, catching "
            "12 passes for 215 yards and 3 touchdowns before leaving with an injury. Alabama's "
            "offense was unstoppable, putting up 35 first-half points. Mac Jones threw for 464 "
            "yards as the Tide cruised to their sixth national title under Nick Saban."
        ),
        "key_stats": {"margin": 28, "passing_yards": 464, "receiver_yards": 215},
        "tags": ["championship", "offensive_explosion", "heisman_performance", "dominant"],
    },
    {
        "id": "2021-rose-bowl-semi",
        "game": "Rose Bowl - CFP Semifinal",
        "year": 2021,
        "date": "2021-01-01",
        "teams": {"winner": "Alabama", "loser": "Notre Dame"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 31, "loser": 14},
        "venue": "AT&T Stadium",
        "narrative": (
            "Alabama's offense proved too much for Notre Dame's defense. The Tide jumped out "
            "to a 21-7 halftime lead and never looked back. Mac Jones threw for 297 yards, and "
            "Najee Harris rushed for 125 yards. Notre Dame's first CFP appearance since 2018 "
            "ended in disappointment."
        ),
        "key_stats": {"margin": 17, "rushing_yards": 125, "first_half_lead": 14},
        "tags": ["semifinal", "comfortable_win", "dominant_half"],
    },
    {
        "id": "2021-sugar-bowl-semi",
        "game": "Sugar Bowl - CFP Semifinal",
        "year": 2021,
        "date": "2021-01-01",
        "teams": {"winner": "Ohio State", "loser": "Clemson"},
        "seeds": {"winner": 3, "loser": 2},
        "score": {"winner": 49, "loser": 28},
        "venue": "Mercedes-Benz Superdome",
        "narrative": (
            "Ohio State ended Clemson's CFP dominance with a 49-28 rout. Justin Fields threw "
            "for 385 yards and 6 touchdowns despite taking a heavy hit that briefly knocked him "
            "out of the game. The Buckeyes' offense was unstoppable, and their defense picked "
            "off Trevor Lawrence twice."
        ),
        "key_stats": {"margin": 21, "passing_tds": 6, "interceptions": 2},
        "tags": ["semifinal", "revenge", "qb_heroics", "statement_win"],
    },
    # 2019-20 CFP
    {
        "id": "2020-cfp-championship",
        "game": "CFP National Championship",
        "year": 2020,
        "date": "2020-01-13",
        "teams": {"winner": "LSU", "loser": "Clemson"},
        "seeds": {"winner": 1, "loser": 3},
        "score": {"winner": 42, "loser": 25},
        "venue": "Mercedes-Benz Superdome",
        "narrative": (
            "Joe Burrow capped the greatest season in college football history, leading LSU to "
            "a 15-0 record and national championship. Burrow threw for 463 yards and 5 "
            "touchdowns, adding a rushing TD for good measure. The Tigers scored 28 unanswered "
            "points in the second half to pull away from Clemson."
        ),
        "key_stats": {
            "margin": 17,
            "passing_yards": 463,
            "passing_tds": 5,
            "undefeated": True,
        },
        "tags": ["championship", "historic_season", "qb_masterpiece", "undefeated"],
    },
    {
        "id": "2020-peach-bowl-semi",
        "game": "Peach Bowl - CFP Semifinal",
        "year": 2020,
        "date": "2019-12-28",
        "teams": {"winner": "LSU", "loser": "Oklahoma"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 63, "loser": 28},
        "venue": "Mercedes-Benz Stadium",
        "narrative": (
            "Joe Burrow threw 7 touchdown passes - a CFP record - as LSU demolished Oklahoma. "
            "The Tigers scored on 8 of their first 9 possessions, and Burrow completed 29 of 39 "
            "passes for 493 yards. It was the most points ever scored in a CFP semifinal. Jalen "
            "Hurts threw for 217 yards in a losing effort."
        ),
        "key_stats": {"margin": 35, "passing_tds": 7, "cfp_record": True},
        "tags": ["semifinal", "record_breaking", "blowout", "offensive_explosion"],
    },
    {
        "id": "2020-fiesta-bowl-semi",
        "game": "Fiesta Bowl - CFP Semifinal",
        "year": 2020,
        "date": "2019-12-28",
        "teams": {"winner": "Clemson", "loser": "Ohio State"},
        "seeds": {"winner": 3, "loser": 2},
        "score": {"winner": 29, "loser": 23},
        "venue": "State Farm Stadium",
        "narrative": (
            "Clemson overcame a 16-0 halftime deficit to stun Ohio State. Trevor Lawrence threw "
            "for 259 yards, but it was the Tigers' defense that made the difference, forcing "
            "two pivotal turnovers in the second half. A controversial targeting call on a "
            "would-be Ohio State interception proved decisive as Clemson rallied."
        ),
        "key_stats": {"comeback": 16, "margin": 6, "turnovers_forced": 2},
        "tags": ["semifinal", "comeback", "controversial", "defensive_turnaround"],
    },
    # 2018-19 CFP
    {
        "id": "2019-cfp-championship",
        "game": "CFP National Championship",
        "year": 2019,
        "date": "2019-01-07",
        "teams": {"winner": "Clemson", "loser": "Alabama"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 44, "loser": 16},
        "venue": "Levi's Stadium",
        "narrative": (
            "Clemson delivered a shocking blowout of Alabama to win the national title. Trevor "
            "Lawrence, just a freshman, threw for 347 yards and 3 touchdowns. The Tigers' "
            "defense held Tua Tagovailoa to just 22-of-34 passing with 2 interceptions. It was "
            "Alabama's worst loss under Nick Saban."
        ),
        "key_stats": {"margin": 28, "freshman_qb": True, "interceptions": 2},
        "tags": ["championship", "upset", "freshman_qb", "statement_win", "dynasty_stunner"],
    },
    {
        "id": "2019-cotton-bowl-semi",
        "game": "Cotton Bowl - CFP Semifinal",
        "year": 2019,
        "date": "2018-12-29",
        "teams": {"winner": "Clemson", "loser": "Notre Dame"},
        "seeds": {"winner": 2, "loser": 3},
        "score": {"winner": 30, "loser": 3},
        "venue": "AT&T Stadium",
        "narrative": (
            "Clemson's defense completely shut down Notre Dame's offense in a dominant "
            "semifinal performance. The Tigers allowed just 88 total yards in the first half "
            "and cruised to a 30-3 victory. Trevor Lawrence threw for 327 yards as Clemson "
            "advanced to face Alabama."
        ),
        "key_stats": {"margin": 27, "yards_allowed": 248, "dominant_defense": True},
        "tags": ["semifinal", "defensive_shutout", "dominant", "statement"],
    },
    {
        "id": "2019-orange-bowl-semi",
        "game": "Orange Bowl - CFP Semifinal",
        "year": 2019,
        "date": "2018-12-29",
        "teams": {"winner": "Alabama", "loser": "Oklahoma"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 45, "loser": 34},
        "venue": "Hard Rock Stadium",
        "narrative": (
            "Tua Tagovailoa outdueled Kyler Murray in an epic quarterback showdown. Both "
            "Heisman finalists lit up the scoreboard, with Tua throwing for 318 yards and 4 "
            "touchdowns. Alabama's defense made decisive stops late to hold off Oklahoma's "
            "high-powered offense."
        ),
        "key_stats": {"combined_points": 79, "passing_yards": 318, "heisman_duel": True},
        "tags": ["semifinal", "shootout", "heisman_qb_duel", "offensive_showcase"],
    },
    # 2017-18 CFP
    {
        "id": "2018-cfp-championship",
        "game": "CFP National Championship",
        "year": 2018,
        "date": "2018-01-08",
        "teams": {"winner": "Alabama", "loser": "Georgia"},
        "seeds": {"winner": 4, "loser": 3},
        "score": {"winner": 26, "loser": 23},
        "venue": "Mercedes-Benz Stadium",
        "narrative": (
            "In one of the most dramatic finishes in championship history, Tua Tagovailoa came "
            "off the bench to lead Alabama to an overtime victory. Trailing 13-0 at halftime, "
            "Saban benched Jalen Hurts for the true freshman. Tua threw the game-winning "
            "41-yard touchdown to DeVonta Smith in overtime. It was Alabama's 17th national "
            "title."
        ),
        "key_stats": {"overtime": True, "comeback": 13, "freshman_heroics": True},
        "tags": ["championship", "overtime", "qb_switch", "instant_classic", "iconic_finish"],
    },
    {
        "id": "2018-rose-bowl-semi",
        "game": "Rose Bowl - CFP Semifinal",
        "year": 2018,
        "date": "2018-01-01",
        "teams": {"winner": "Georgia", "loser": "Oklahoma"},
        "seeds": {"winner": 3, "loser": 2},
        "score": {"winner": 54, "loser": 48},
        "venue": "Rose Bowl",
        "narrative": (
            "Georgia and Oklahoma played one of the greatest games in college football history. "
            "The Bulldogs won in double overtime after Baker Mayfield and Sony Michel traded "
            "haymakers all game. Michel's 27-yard TD run in the second overtime won it. The "
            "game featured 1,170 total yards of offense."
        ),
        "key_stats": {"double_overtime": True, "combined_points": 102, "total_yards": 1170},
        "tags": [
            "semifinal",
            "double_overtime",
            "shootout",
            "greatest_rose_bowl",
            "instant_classic",
        ],
    },
    {
        "id": "2018-sugar-bowl-semi",
        "game": "Sugar Bowl - CFP Semifinal",
        "year": 2018,
        "date": "2018-01-01",
        "teams": {"winner": "Alabama", "loser": "Clemson"},
        "seeds": {"winner": 4, "loser": 1},
        "score": {"winner": 24, "loser": 6},
        "venue": "Mercedes-Benz Superdome",
        "narrative": (
            "Alabama's defense dominated Clemson in a CFP rematch. The Tide held the Tigers to "
            "just 188 total yards and forced two turnovers. Jalen Hurts threw for 120 yards and "
            "rushed for 47 as Alabama controlled the game from start to finish."
        ),
        "key_stats": {"margin": 18, "yards_allowed": 188, "turnovers": 2},
        "tags": ["semifinal", "defensive_masterpiece", "revenge", "dominant"],
    },
    # 2016-17 CFP
    {
        "id": "2017-cfp-championship",
        "game": "CFP National Championship",
        "year": 2017,
        "date": "2017-01-09",
        "teams": {"winner": "Clemson", "loser": "Alabama"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 35, "loser": 31},
        "venue": "Raymond James Stadium",
        "narrative": (
            "Deshaun Watson threw a 2-yard touchdown pass to Hunter Renfrow with 1 second "
            "remaining to give Clemson its first national title since 1981. Watson finished "
            "with 420 passing yards against Alabama's vaunted defense. The game-winning drive "
            "covered 68 yards in 4 plays, with Watson delivering under intense pressure."
        ),
        "key_stats": {"margin": 4, "passing_yards": 420, "game_winning_drive": True},
        "tags": ["championship", "last_second_win", "qb_heroics", "iconic_finish", "classic"],
    },
    {
        "id": "2017-fiesta-bowl-semi",
        "game": "Fiesta Bowl - CFP Semifinal",
        "year": 2017,
        "date": "2016-12-31",
        "teams": {"winner": "Clemson", "loser": "Ohio State"},
        "seeds": {"winner": 2, "loser": 3},
        "score": {"winner": 31, "loser": 0},
        "venue": "University of Phoenix Stadium",
        "narrative": (
            "Clemson delivered the first shutout in CFP history, blanking Ohio State 31-0. The "
            "Tigers forced three turnovers and held the Buckeyes to just 215 total yards. "
            "Deshaun Watson threw for 259 yards and rushed for 57 more in the dominant victory."
        ),
        "key_stats": {"shutout": True, "margin": 31, "turnovers_forced": 3},
        "tags": ["semifinal", "shutout", "cfp_first", "dominant", "defensive_masterpiece"],
    },
    {
        "id": "2017-peach-bowl-semi",
        "game": "Peach Bowl - CFP Semifinal",
        "year": 2017,
        "date": "2016-12-31",
        "teams": {"winner": "Alabama", "loser": "Washington"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 24, "loser": 7},
        "venue": "Georgia Dome",
        "narrative": (
            "Alabama's defense smothered Washington's high-powered offense. The Tide held the "
            "Huskies to just 194 total yards and forced two turnovers. Jalen Hurts threw for 57 "
            "yards and rushed for 50 in a game Alabama controlled from start to finish."
        ),
        "key_stats": {"margin": 17, "yards_allowed": 194, "defensive_dominance": True},
        "tags": ["semifinal", "defensive_dominance", "comfortable_win"],
    },
    # 2015-16 CFP
    {
        "id": "2016-cfp-championship",
        "game": "CFP National Championship",
        "year": 2016,
        "date": "2016-01-11",
        "teams": {"winner": "Alabama", "loser": "Clemson"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 45, "loser": 40},
        "venue": "University of Phoenix Stadium",
        "narrative": (
            "Alabama edged Clemson in an instant classic to win its fourth national title in "
            "seven years. Derrick Henry rushed for 158 yards and 3 touchdowns. Deshaun Watson "
            "threw for 405 yards in a losing effort, but an onside kick recovery sealed the win "
            "for the Tide. It set up a legendary rematch the following year."
        ),
        "key_stats": {"margin": 5, "combined_points": 85, "rushing_yards": 158},
        "tags": [
            "championship",
            "classic",
            "offensive_showcase",
            "onside_kick",
            "close_finish",
        ],
    },
    {
        "id": "2016-orange-bowl-semi",
        "game": "Orange Bowl - CFP Semifinal",
        "year": 2016,
        "date": "2015-12-31",
        "teams": {"winner": "Clemson", "loser": "Oklahoma"},
        "seeds": {"winner": 1, "loser": 4},
        "score": {"winner": 37, "loser": 17},
        "venue": "Sun Life Stadium",
        "narrative": (
            "Deshaun Watson dazzled in his first CFP game, throwing for 187 yards and rushing "
            "for 145 more with two touchdowns. Clemson's defense forced three turnovers as the "
            "Tigers cruised past Oklahoma. It was Clemson's first playoff win in program "
            "history."
        ),
        "key_stats": {"margin": 20, "dual_threat_yards": 332, "turnovers_forced": 3},
        "tags": ["semifinal", "dual_threat_qb", "program_first", "comfortable_win"],
    },
    {
        "id": "2016-cotton-bowl-semi",
        "game": "Cotton Bowl - CFP Semifinal",
        "year": 2016,
        "date": "2015-12-31",
        "teams": {"winner": "Alabama", "loser": "Michigan State"},
        "seeds": {"winner": 2, "loser": 3},
        "score": {"winner": 38, "loser": 0},
        "venue": "AT&T Stadium",
        "narrative": (
            "Alabama pitched the first shutout in CFP history (later matched by Clemson), "
            "demolishing Michigan State 38-0. Derrick Henry rushed for 75 yards and 2 "
            "touchdowns, while the defense held the Spartans to just 239 total yards. Connor "
            "Cook was sacked three times in the Spartans' worst loss of the season."
        ),
        "key_stats": {"shutout": True, "margin": 38, "sacks": 3},
        "tags": ["semifinal", "shutout", "cfp_first", "dominant", "defensive_masterpiece"],
    },
    # 2014-15 CFP (Inaugural Season)
    {
        "id": "2015-cfp-championship",
        "game": "CFP National Championship",
        "year": 2015,
        "date": "2015-01-12",
        "teams": {"winner": "Ohio State", "loser": "Oregon"},
        "seeds": {"winner": 4, "loser": 2},
        "score": {"winner": 42, "loser": 20},
        "venue": "AT&T Stadium",
        "narrative": (
            "Ohio State capped an improbable run to win the first-ever College Football "
            "Playoff. The Buckeyes, who entered as the 4-seed after losing their starting "
            "quarterback, rode Ezekiel Elliott's 246 rushing yards to the title. Cardale Jones, "
            "the third-string QB, threw for 242 yards as Ohio State became the first CFP "
            "champion."
        ),
        "key_stats": {"margin": 22, "rushing_yards": 246, "third_string_qb": True},
        "tags": [
            "championship",
            "inaugural_cfp",
            "underdog",
            "rushing_dominance",
            "improbable",
        ],
    },
    {
        "id": "2015-sugar-bowl-semi",
        "game": "Sugar Bowl - CFP Semifinal",
        "year": 2015,
        "date": "2015-01-01",
        "teams": {"winner": "Ohio State", "loser": "Alabama"},
        "seeds": {"winner": 4, "loser": 1},
        "score": {"winner": 42, "loser": 35},
        "venue": "Mercedes-Benz Superdome",
        "narrative": (
            "In the first CFP semifinal ever, 4-seed Ohio State shocked 1-seed Alabama. The "
            "Buckeyes erased a 21-6 deficit behind Ezekiel Elliott's 230 rushing yards. Cardale "
            "Jones, starting just his second career game, threw for 243 yards in the upset that "
            "launched Ohio State's championship run."
        ),
        "key_stats": {"margin": 7, "comeback": 15, "rushing_yards": 230},
        "tags": ["semifinal", "inaugural_cfp", "upset", "comeback", "historic"],
    },
    {
        "id": "2015-rose-bowl-semi",
        "game": "Rose Bowl - CFP Semifinal",
        "year": 2015,
        "date": "2015-01-01",
        "teams": {"winner": "Oregon", "loser": "Florida State"},
        "seeds": {"winner": 2, "loser": 3},
        "score": {"winner": 59, "loser": 20},
        "venue": "Rose Bowl",
        "narrative": (
            "Oregon's high-powered offense destroyed Florida State's 29-game winning streak. "
            "Marcus Mariota threw for 338 yards and 2 touchdowns while rushing for 2 more. The "
            "Ducks scored on their first five possessions as Jameis Winston threw 2 "
            "interceptions in FSU's worst loss in years."
        ),
        "key_stats": {"margin": 39, "streak_ended": 29, "first_half_points": 32},
        "tags": [
            "semifinal",
            "inaugural_cfp",
            "blowout",
            "streak_buster",
            "offensive_explosion",
        ],
    },
    # Notable BCS Era Games (2006-2014)
    {
        "id": "2014-bcs-championship",
        "game": "BCS National Championship",
        "year": 2014,
        "date": "2014-01-06",
        "teams": {"winner": "Florida State", "loser": "Auburn"},
        "seeds": {"winner": 1, "loser": 2},
        "score": {"winner": 34, "loser": 31},
        "venue": "Rose Bowl",
        "narrative": (
            "The final BCS Championship Game went down to the wire. Jameis Winston drove "
            "Florida State 80 yards in the final two minutes, hitting Kelvin Benjamin for the "
            "game-winning touchdown with 13 seconds left. Auburn's miracle season ended just "
            "short after the 'Kick Six' had gotten them there."
        ),
        "key_stats": {"margin": 3, "game_winning_drive": True, "last_bcs_game": True},
        "tags": [
            "championship",
            "bcs_finale",
            "last_second_win",
            "classic",
            "game_winning_drive",
        ],
    },
    {
        "id": "2013-bcs-championship",
        "game": "BCS National Championship",
        "year": 2013,
        "date": "2013-01-07",
        "teams": {"winner": "Alabama", "loser": "Notre Dame"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 42, "loser": 14},
        "venue": "Sun Life Stadium",
        "narrative": (
            "Alabama dominated Notre Dame from start to finish in a BCS Championship rout. AJ "
            "McCarron threw for 264 yards and 4 touchdowns, while Eddie Lacy rushed for 140 "
            "yards. The Tide built a 28-0 halftime lead and cruised to their third title in "
            "four years."
        ),
        "key_stats": {"margin": 28, "first_half_shutout": True, "dynasty_peak": True},
        "tags": ["championship", "blowout", "dynasty", "dominant"],
    },
    {
        "id": "2012-bcs-championship",
        "game": "BCS National Championship",
        "year": 2012,
        "date": "2012-01-09",
        "teams": {"winner": "Alabama", "loser": "LSU"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 21, "loser": 0},
        "venue": "Mercedes-Benz Superdome",
        "narrative": (
            "In a defensive struggle, Alabama shut out LSU in a BCS Championship rematch. The "
            "Tide avenged their only regular-season loss with a dominant defensive performance. "
            "Trent Richardson rushed for 96 yards as Alabama held LSU to just 92 total yards. "
            "It remains the only shutout in BCS Championship history."
        ),
        "key_stats": {"shutout": True, "yards_allowed": 92, "rematch": True},
        "tags": ["championship", "shutout", "defensive_classic", "rematch", "historic"],
    },
    {
        "id": "2006-rose-bowl",
        "game": "Rose Bowl - BCS Championship",
        "year": 2006,
        "date": "2006-01-04",
        "teams": {"winner": "Texas", "loser": "USC"},
        "seeds": {"winner": 2, "loser": 1},
        "score": {"winner": 41, "loser": 38},
        "venue": "Rose Bowl",
        "narrative": (
            "Vince Young delivered one of the greatest performances in championship history, "
            "rushing for 200 yards and the game-winning touchdown with 19 seconds left. Texas "
            "came back from a 12-point fourth-quarter deficit to dethrone USC and end their "
            "34-game winning streak. Young's fourth-down scramble remains one of the most "
            "iconic plays in college football history."
        ),
        "key_stats": {
            "margin": 3,
            "comeback": 12,
            "rushing_yards_qb": 200,
            "streak_ended": 34,
        },
        "tags": [
            "championship",
            "greatest_game",
            "qb_heroics",
            "comeback",
            "iconic_play",
            "classic",
        ],
    },
]


def load_historical_games() -> list[Game]:
    """Validate and return all 36 historical games as immutable Game models."""
    return [Game.model_validate(raw) for raw in HISTORICAL_GAMES]


def filter_games(
    games: Sequence[Game],
    *,
    year: int | None = None,
    tag: str | None = None,
    team: str | None = None,
    overtime_only: bool = False,
) -> list[Game]:
    """Filter validated Game models by year, tag, team substring, or overtime flag."""
    team_query = team.strip().lower() if team else None
    tag_query = tag.strip().lower() if tag else None

    matched: list[Game] = []
    for game in games:
        if year is not None and game.year != year:
            continue
        if tag_query and tag_query not in [t.lower() for t in game.tags]:
            continue
        if team_query and (
            team_query not in game.teams.winner.lower()
            and team_query not in game.teams.loser.lower()
        ):
            continue
        if overtime_only and not (game.key_stats and game.key_stats.overtime):
            continue
        matched.append(game)
    return matched


def get_games_by_tag(tag: str, games: Sequence[Game] | None = None) -> list[Game]:
    """Return all games matching a specific tag."""
    catalog = games if games is not None else load_historical_games()
    return filter_games(catalog, tag=tag)


def get_games_by_year(year: int, games: Sequence[Game] | None = None) -> list[Game]:
    """Return all games played in a specific year."""
    catalog = games if games is not None else load_historical_games()
    return filter_games(catalog, year=year)


def get_games_by_team(team: str, games: Sequence[Game] | None = None) -> list[Game]:
    """Return all games featuring a specific team (case-insensitive)."""
    catalog = games if games is not None else load_historical_games()
    return filter_games(catalog, team=team)
