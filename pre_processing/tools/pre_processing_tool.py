from datetime import datetime


def strip(exp_date, type):
        stripped_date = datetime.strptime(exp_date, "%Y-%m-%d %H:%M:%S")
        if type == "date":
            return stripped_date.date()
        elif type == "time":
            return stripped_date.time()
        elif type == "year":
            return stripped_date.year
        elif type == "month":
            return stripped_date.month
        elif type == "day":
            return stripped_date.day

def strip_srch(exp_date, type):
    stripped_date = datetime.strptime(exp_date, "%Y-%m-%d")
    if type == "date":
        return stripped_date.year
    elif type == "time":
        return stripped_date.time()
    elif type == "year":
        return stripped_date.year
    elif type == "month":
        return stripped_date.month
    elif type == "day":
        return stripped_date.day



def date_subtract(date1,date2):
    stripped_date1 = datetime.strptime(date1, "%Y-%m-%d")
    stripped_date2 = datetime.strptime(date2, "%Y-%m-%d")
    date_difference = stripped_date2 - stripped_date1
    return date_difference


def create_month_bins(month):
    if month <=3:
        return  1
    elif month > 3 and month <=6:
        return  2
    elif month >6 and month <=9:
        return 3
    elif month >9 and month <=12:
        return 4

def check_season(season_bin,season):

    if season == "winter":
        if season_bin == 1:
            return 1
        else:
            return 0
    elif season == "spring":
        if season_bin == 2:
            return 1
        else:
            return 0
    elif season == "summer":
        if season_bin == 3:
            return 1
        else:
            return 0
    elif season == "fall":
        if season_bin == 4:
            return 1
        else:
            return 0

def weekend_check(day):
    if day == 6 or day == 5:
        return 1
    else:
        return 0

def check_time(time, day_time):
    day_bins = ["6:00:00","10:00:00","14:00:00:","16:00:00","20:00:00","24:00:00"]

    import datetime

    time_1 = datetime.time(0, 0, 0)
    time_2 = datetime.time(6, 0, 0)
    time_3 = datetime.time(8, 0, 0)
    time_4 = datetime.time(10, 0, 0)
    time_5 = datetime.time(14, 0, 0)
    time_6 = datetime.time(16, 0, 0)
    time_7 = datetime.time(20, 0, 0)

    if day_time == "late_night":
        if time >= time_1 and time < time_2:
            return 1
        else:
            return 0

    if day_time == "early_morning":
        if time >= time_2 and time < time_3:
            return 1
        else:
            return 0

    if day_time == "morning":
        if time >= time_3 and time < time_4:
            return 1
        else:
            return 0

    if day_time == "mid_day":
        if time >= time_4 and time < time_5:
            return 1
        else:
            return 0

    if day_time == "afternoon":
        if time >= time_5 and time < time_6:
            return 1
        else:
            return 0

    if day_time == "evening":
        if time >= time_6 and time < time_7:
            return 1
        else:
            return 0

    if day_time == "night":
        if time >= time_7:
            return 1
        else:
            return 0






