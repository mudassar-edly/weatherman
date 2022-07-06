import pandas as pd
import numpy as np
import os
import sys
from glob import glob

class Reading:
    """ The data-structure to hold the readings.
    """

    def __init__(self,  month, year,
                        max_temp, 
                        min_temp, 
                        mean_humidity):
        """ Intialize an instance of the Reading class.

        Args:
            month (string):                The full name of the Month for which this column of data is added
            year  (string):                The year for which the data is stored (e.g. '2004')
            max_temp (numpy.ndarray):      A 2d array of shape (31,1) containing values of maximum daily temperature.
            min_temp (numpy.ndarray):      A 2d array of shape (31,1) containing values of minimum daily temperature.
            mean_humidity (numpy.ndarray): A 2d array of shape (31,1) containing values of mean daily humidity.

        Instance attributes:
            idx_to_month (dict): A mapping of a column index to the full name of the month of which it contains the data (e.g. {0:March})
        """
        self.max_temps     = max_temp
        self.min_temps     = min_temp
        self.mean_humidity = mean_humidity
        self.idx_to_month  = {0:month}
        self.year          = year

    def append  (self,  month,
                        max_temp, 
                        min_temp, 
                        mean_humidity):
        """ Append data to an instance of the Reading class. Rows stay as days, columns represent months.

        Args:
            month (string):                The full name of the Month of the data columns.
            max_temp (numpy.ndarray):      A 2d array of shape (31,1) containing values of maximum daily temperature.
            min_temp (numpy.ndarray):      A 2d array of shape (31,1) containing values of minimum daily temperature.
            mean_humidity (numpy.ndarray): A 2d array of shape (31,1) containing values of mean daily humidity.

        Instance attributes:
            idx_to_month (dict): A mapping of a column index to the full name of the month of which it contains the data (e.g. {1:August})
        """

        self.idx_to_month[self.max_temps.shape[1]] = month
        self.max_temps     = np.hstack([self.max_temps,     max_temp     ])
        self.min_temps     = np.hstack([self.min_temps,     min_temp     ])
        self.mean_humidity = np.hstack([self.mean_humidity, mean_humidity])


class ParseFilesAndPopulateReadings:
    """A helper class to read data files and extract relevant data from them.

    Returns:
        readings: An instance of Reading class data-structure that contains the relevant data of the specified month and year
    
    Class Attributes:
        __cols_to_extract (list): A list containing the title of the columns to be extracted from the data.
        __num_to_month (dict):    A mapping from month number to month name.
    """
    __cols_to_extract = ['Max TemperatureC', 'Min TemperatureC', ' Mean Humidity'] 
    __num_to_month    = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

    def __call__(self, base_dir, year, month):
        """Reads relavent files and returns data in appropritate format.

        Args:
            base_dir (string): Path to the directory containing the data files.
            year (string):     The year of which the data has to be loaded (e.g. '2015')
            month (None/int):  The month number of the month of which data has to be loaded. 
                               None if all months of the specified year have to be loaded.

        Returns:
            readings: An instance of Reading class data-structure that contains the relevant data of the specified month and year.
        
        Assumptions:
            The datafiles must have a row for each day in the month in sorted order. 
            It should not be like [May 1, May 2, May 3, May 6, May 7, May 9, ...]
        """
        month = '*' if month == None else self.__num_to_month[month] # read all months if one month is not specified

        file_paths = glob(os.path.join(base_dir, f'Murree_weather_{year}_{month[:3]}.txt'))
        readings = None
        
        for path in file_paths:
            filename = os.path.split(path)[-1] 
            filename = filename.split('.')[ 0] # remove file extension (.txt)
            month    = filename.split('_')[-1] 

            # get the full name of the month by comparing the initial 3 characters.
            for full_month in self.__num_to_month.values():
                if month == full_month[:3]:
                    month = full_month
                    break

            df = pd.read_csv(path)

            extracted_columns = []
            for col in self.__cols_to_extract:

                # extract a column and typecast it into float so that NaN can also fit in the array.
                data = df[col].to_numpy(dtype=np.float64) 

                # pad the data to 31 days (here an assumption is made that no file has a row missing for any day in the month)
                data = np.pad(data, (0, 31-len(data)), mode='constant', constant_values=np.nan).reshape((-1,1))
                
                extracted_columns.append(data)

            if readings == None:
                readings = Reading(month, year, *extracted_columns)
            else:
                readings . append (month, *extracted_columns)

        return readings


class Result:
    """ A Data-Structure to store the results of calculations done on the data.
    """
    def __init__(self,  option, year,
                        max_temp,          min_temp,          mean_humidity,
                        max_temp_day='',   min_temp_day='',   mean_humidity_day='',
                        max_temp_month='', min_temp_month='', mean_humidity_month=''):
        """Initialize an instance of the Results class

        Args:
            option (string): The command line switch that determines the type of calculation done on the data.
            year (string): The year of which the data is loaded (e.g. '2005')
            max_temp (float/numpy.ndarray): The highest temperature(s) in the data.
            min_temp (float/numpy.ndarray): The lowest temperature(s) in the data.
            mean_humidity (float): The statistical average of the humidity values in the data.
            max_temp_day (str, optional): The day of the reading value (e.g. 5 of May). Defaults to ''.
            min_temp_day (str, optional): The day of the reading value (e.g. 5 of May). Defaults to ''.
            mean_humidity_day (str, optional): The day of the reading value (e.g. 5 of May). Defaults to ''.
            max_temp_month (str, optional): The month of the reading value (e.g. May). Defaults to ''.
            min_temp_month (str, optional): The month of the reading value (e.g. May). Defaults to ''.
            mean_humidity_month (str, optional): The month of the reading value (e.g. May). Defaults to ''.
        """
        self.max_temp       = max_temp
        self.max_temp_day   = max_temp_day
        self.max_temp_month = max_temp_month
        
        self.min_temp       = min_temp
        self.min_temp_day   = min_temp_day
        self.min_temp_month = min_temp_month

        self.mean_humidity       = mean_humidity
        self.mean_humidity_day   = mean_humidity_day
        self.mean_humidity_month = mean_humidity_month

        self.option = option
        self.year   = year 

class Calculate:
    """ A helper class to perform calculations on the data.
    """
    def __call__(self,readings,option):
        """The function that actually performs the calculations e.g. find minimum, maximum, mean etc.

        Args:
            readings (Reading): An instance of the Reading class Data-structure that contains the data to be processed.
            option (string): The command line switch that specifies the action to be performed.

        Raises:
            ValueError: A switch value that does not correspond to any calculation

        Returns:
            results: An object of the Result class containing the results of the calculations.
        """
        if option == '-e':

            # find the maximum temperature and use its indexes to figure out the month and date the reading was taken on.
            highest         = np.nanmax   (  readings.max_temps )
            highest_idx     = np.nanargmax(  readings.max_temps )
            highest_col_idx = highest_idx %  readings.max_temps.shape[1]
            highest_row_idx = highest_idx // readings.max_temps.shape[1]

            # find the minimum temperature and use its indexes to figure out the month and date the reading was taken on.
            lowest         = np.nanmin   ( readings.min_temps )
            lowest_idx     = np.nanargmin( readings.min_temps )
            lowest_col_idx = lowest_idx %  readings.min_temps.shape[1]
            lowest_row_idx = lowest_idx // readings.min_temps.shape[1]

            # find the maximum humidity and use its indexes to figure out the month and date the reading was taken on.
            humid         = np.nanmax   ( readings.mean_humidity )
            humid_idx     = np.nanargmax( readings.mean_humidity )
            humid_col_idx = humid_idx  %  readings.mean_humidity.shape[1]
            humid_row_idx = humid_idx  // readings.mean_humidity.shape[1]

            return Result(option, readings.year, max_temp=highest, min_temp=lowest, mean_humidity=humid,

                        # day_num = index + 1
                        max_temp_day      = highest_row_idx + 1, 
                        min_temp_day      = lowest_row_idx  + 1, 
                        mean_humidity_day = humid_row_idx   + 1,

                        max_temp_month      = readings.idx_to_month[highest_col_idx], 
                        min_temp_month      = readings.idx_to_month[ lowest_col_idx], 
                        mean_humidity_month = readings.idx_to_month[  humid_col_idx])

        elif option == '-a':
            assert readings.max_temps.shape[1] == 1 # Data loaded for only one month
            avg_highest = np.nanmean( readings.max_temps )
            avg_lowest  = np.nanmean( readings.min_temps )
            avg_humid   = np.nanmean( readings.mean_humidity )

            return Result(option, readings.year, max_temp=avg_highest, min_temp=avg_lowest, mean_humidity=avg_humid)

        elif option == '-c':
            assert readings.max_temps.shape[1] == 1 # Data loaded for only one month
            return Result(option, readings.year, 
                                    max_temp=np.squeeze(readings.max_temps), # the data was 2d - rows = days, cols = months but since there is only one month, squeeze to 1d.
                                    min_temp=np.squeeze(readings.min_temps), 
                                    mean_humidity=None, 
                                    max_temp_month=readings.idx_to_month[0], # store the value of the month somewhere in the results datastructure as well.
                                    min_temp_month=readings.idx_to_month[0]  )

        else:
            raise ValueError(f'"{option}" is not a valid option')

class Colors:
    RED  = '\033[91m'
    BLUE = '\033[96m'
    ENDC = '\033[0m'

class CreateReport:
    """ A helper class to synthesize user understandable reports given cirtain results.
    """

    def __call__(self, results):
        """The function that actually generates the reports from the data.

        Args:
            results (Result): An instance of Results data-structure that contains the results of the calculations performed.

        Raises:
            ValueError: Invalid switch provided in the command line arguments.

        Returns:
            String: A string containing the text of the report.
        """
        if results.option == '-e':
            report  = f'Highest: {results.max_temp:.0f}C on {results.max_temp_month} {results.max_temp_day}\n'
            report +=  f'Lowest: {results.min_temp:.0f}C on {results.min_temp_month} {results.min_temp_day}\n'
            report += f'Humidity: {results.mean_humidity:.0f}% on {results.mean_humidity_month} {results.mean_humidity_day}\n'
            
        elif results.option == '-a':
            report  =  f'Average Highest Temp: {results.max_temp:.0f}C\n'
            report +=   f'Average Lowest Temp: {results.min_temp:.0f}C\n'
            report += f'Average Mean Humidity: {results.mean_humidity:.0f}%\n'

        elif results.option == '-c':
            report = f'{results.max_temp_month} {results.year}\n' # May 2011
            for i in range(len(results.max_temp)):
                high, low = results.max_temp[i], results.min_temp[i]
                day = i + 1

                # if not bonus
                if not np.isnan(high):
                    report += f'{day:02d} {Colors.RED }{"+"* int(high) }{Colors.ENDC} {high:.0f}C\n' # 01 ++++++ 6C
                if not np.isnan(low ):
                    report += f'{day:02d} {Colors.BLUE}{"+"* int(low)  }{Colors.ENDC} {low :.0f}C\n' # 01 +++ 3C

                # if bonus
                # if not (np.isnan(high) or np.isnan(low)):
                #     report += f'{day:02d} {Colors.BLUE}{"+"* int(low)  }{Colors.RED }{"+"* int(high) }{Colors.ENDC} {high:.0f}C - {low:.0f}C\n'

        else:
            raise ValueError(f'"{results.option}" is not a valid option')

        return report



def main():
    """ Assembles the entire Program. Takes command line options and run the relevant calculations and prints the resulting reports.

    Raises:
        ValueError: invalid number of arguments
        ValueError: not a valid year  for non-numeric year
        ValueError: not a valid month for non numeric month
        ValueError: not a valid year/month format argument
        ValueError: not a valid option/switch
        ValueError: found no datafile for specified year/month
    """
    dir = sys.argv[1] # /User/Mudassariqbal/Downloads/weatherfiles

    for i in range(2,len(sys.argv),2):
        try: 
            option = sys.argv[i]
            val = sys.argv[i+1]
        except:
            raise ValueError('invalid number of arguments')

        # find day/month when min/max temperature occured or max humidity occured
        if option == '-e': 
            if not val.isdigit():
                raise ValueError(f'"{val}" is not a valid year')
            
            year, month = val, None # None means evaluate data of all months

        # Analyse data of a specific month
        # -a : find average max/min temperatures
        # -c : find average humidity
        elif option in ['-a','-c']:
            vals = val.split('/')
            if len(vals) != 2:
                raise ValueError(f'"{vals   }" is not a valid "year/month" input')
            if not vals[0].isdigit():
                raise ValueError(f'"{vals[0]}" is not a valid year')
            if not vals[1].isdigit():
                raise ValueError(f'"{vals[0]}" is not a valid month')

            year, month = vals[0], int(vals[1])

        else:
            raise ValueError(f'"{option}" is not a valid option')


        parse = ParseFilesAndPopulateReadings()
        readings = parse(dir, year, month)

        if readings is None:
            raise ValueError(f'Found no data file for "{val}"')

        calc = Calculate()
        results = calc(readings, option)

        cr = CreateReport()
        report = cr(results)

        print(report)

main()