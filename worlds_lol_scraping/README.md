# League of Legends Data Scraping
This project focuses on web scraping to gather data from a League of Legends championship. The goal is to extract comprehensive information about matches, teams, bans, picks, and player rosters. By scraping this data, we can gain insights into the tournament and perform further analysis.
# Functions
The project includes the following functions:

## 1. scraping
This function performs web scraping on a specific web page related to the match history of the League of Legends championship. It retrieves the desired table by searching for its tag and specific class name. The function then iterates through the table rows, extracting information such as the date, patch, blue team, red team, winner, bans (if applicable), picks, and player rosters. The extracted data is stored in a DataFrame.

## 2. df_organize
The df_organize function takes the DataFrame obtained from the scraping process and organizes the data into structured columns. Depending on the specific end point, which represents the championship being scraped, the function handles variations in the table structure. It separates data such as bans (up to five bans per team), picks (up to five picks per team), and player rosters into separate columns. This ensures the data is well-organized and ready for further analysis.

## 3. scrape_table
This function serves as a convenience function that encapsulates the calling of the scraping and df_organize functions. It takes the name of the end point (representing the championship) as input, performs web scraping using the scraping function, and then organizes the scraped data using the df_organize function. In case of any errors during the process, an exception will be raised.

# Purpose
The purpose of this project is to enable the extraction and organization of comprehensive data from League of Legends championships. By utilizing the provided functions, users can scrape match history tables, transform the raw data into a structured format, and leverage it for various analytical purposes, such as studying team performance, analyzing champion preferences, and identifying player trends.

Please note that web scraping should be performed responsibly, adhering to the website's terms of service and legal requirements.







