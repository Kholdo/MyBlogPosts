import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

try:
    connection=mysql.connector.connect(host='yourhost_maybe_localhost',
                                        database='nameofyourdatabase',
                                        user='youruser',
                                        password='yourpassword')
    cursor=connection.cursor()

    sql_query="""SELECT temperatura, timestamp FROM th_probe"""

    cursor.execute(sql_query)
    result=cursor.fetchall()

    header=['temperatura, timestamp']
    data=pd.Dataframe(result, columns=header)

    data.plot(kind='line', x='timestamp', y='temperatura', color='blue')
    plt.show()

except mysql.connector.Error as error:
    print("La conexion con la base de datos ha fallado {}".format(error))
