def insertQuery(data_list):
    import mysql.connector
    from datetime import datetime

    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    data_list.append(formatted_date)

    try:
        connection = mysql.connector.connect(host='yourhost_maybe_localhost',
                                            database='nameofyourdatabase',
                                            user='youruser',
                                            password='yourpassword')
        cursor = connection.cursor()
        sql_query = """INSERT INTO `th_probe`
                        (`humedad_abs`, `humedad_abs_max`, `humedad_abs_min`,
                        `humedad_rel`, `humedad_rel_max`, `humedad_rel_min`,
                        `punto_rocio`, `punto_rocio_max`, `punto_rocio_min`,
                        `temperaura`, `temperatura_max`, `temperatura_min`,
                        `timestamp`) VALUES (%s, %s, %s,
                                            %s, %s, %s, 
                                            %s, %s, %s
                                            %s, %s, %s,
                                            %s)"""
        cursor.execute(sql_query, data_list)
        connection.commit()
        print("Datos grabados correctamente")
    except mysql.connector.Error as error:
        print("La insercion ha fallado. Error {}".format(error))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Conexion con MySQL cerrada")
