from b__mode_counter import csv_mode_preprocessingORIG_DATA = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/z_2__partial_16.csv'TRAINING_DATA = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/training_16by16.csv'TESTING_DATA = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16.csv'TESTING_DATA0 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_0.csv'TESTING_DATA1 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_1.csv'TESTING_DATA2 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_2.csv'TESTING_DATA3 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_3.csv'TESTING_DATA4 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_4.csv'TESTING_DATA5 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_5.csv'TESTING_DATA6 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_6.csv'TESTING_DATA7 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_7.csv'TESTING_DATA8 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_8.csv'TESTING_DATA9 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_9.csv'TESTING_DATA10 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_10.csv'TESTING_DATA11 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_11.csv'TESTING_DATA12 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_12.csv'TESTING_DATA13 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_13.csv'TESTING_DATA14 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_14.csv'TESTING_DATA15 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_15.csv'TESTING_DATA16 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_16.csv'TESTING_DATA17 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_17.csv'TESTING_DATA18 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_18.csv'TESTING_DATA19 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_19.csv'TESTING_DATA20 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_20.csv'TESTING_DATA21 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_21.csv'TESTING_DATA22 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_22.csv'TESTING_DATA23 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_23.csv'TESTING_DATA24 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_24.csv'TESTING_DATA25 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_25.csv'TESTING_DATA26 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_26.csv'TESTING_DATA27 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_27.csv'TESTING_DATA28 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_28.csv'TESTING_DATA29 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_29.csv'TESTING_DATA30 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_30.csv'TESTING_DATA31 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_31.csv'TESTING_DATA32 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_32.csv'TESTING_DATA33 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_33.csv'TESTING_DATA34 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_34.csv'TESTING_DATA35 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_35.csv'TESTING_DATA36 = '/Users/Pharrell_WANG/PycharmProjects/tf_dp/data/testing_16by16_36.csv'def data_generator(ORIG_DATA, TRAINING_DATA, TESTING_DATA, x_ordered_dict,                   TESTING_DATA0,                   TESTING_DATA1,                   TESTING_DATA2,                   # TESTING_DATA3,                   # TESTING_DATA4,                   # TESTING_DATA5,                   # TESTING_DATA6,                   # TESTING_DATA7,                   # TESTING_DATA8,                   # TESTING_DATA9,                   # TESTING_DATA10,                   # TESTING_DATA11,                   # TESTING_DATA12,                   # TESTING_DATA13,                   # TESTING_DATA14,                   # TESTING_DATA15,                   # TESTING_DATA16,                   # TESTING_DATA17,                   # TESTING_DATA18,                   # TESTING_DATA19,                   # TESTING_DATA20,                   # TESTING_DATA21,                   # TESTING_DATA22,                   # TESTING_DATA23,                   TESTING_DATA24,                   TESTING_DATA25,                   TESTING_DATA26,                   TESTING_DATA27,                   TESTING_DATA28,                   TESTING_DATA29,                   TESTING_DATA30,                   TESTING_DATA31,                   TESTING_DATA32,                   TESTING_DATA33,                   TESTING_DATA34,                   TESTING_DATA35,                   TESTING_DATA36):    with open(ORIG_DATA, 'r') as orig_data, open(TRAINING_DATA, 'w') as training_data, \            open(TESTING_DATA, 'w') as testing_data, \            open(TESTING_DATA0, 'w') as testing_data0, \            open(TESTING_DATA1, 'w') as testing_data1, \            open(TESTING_DATA2, 'w') as testing_data2, \            open(TESTING_DATA24, 'w') as testing_data24, \            open(TESTING_DATA25, 'w') as testing_data25, \            open(TESTING_DATA26, 'w') as testing_data26, \            open(TESTING_DATA27, 'w') as testing_data27, \            open(TESTING_DATA28, 'w') as testing_data28, \            open(TESTING_DATA29, 'w') as testing_data29, \            open(TESTING_DATA30, 'w') as testing_data30, \            open(TESTING_DATA31, 'w') as testing_data31, \            open(TESTING_DATA32, 'w') as testing_data32, \            open(TESTING_DATA33, 'w') as testing_data33, \            open(TESTING_DATA34, 'w') as testing_data34, \            open(TESTING_DATA35, 'w') as testing_data35, \            open(TESTING_DATA36, 'w') as testing_data36:        # open(TESTING_DATA3, 'w') as testing_data3, \        # open(TESTING_DATA4, 'w') as testing_data4, \        # open(TESTING_DATA5, 'w') as testing_data5, \        # open(TESTING_DATA6, 'w') as testing_data6, \        # open(TESTING_DATA7, 'w') as testing_data7, \        # open(TESTING_DATA8, 'w') as testing_data8, \        # open(TESTING_DATA9, 'w') as testing_data9, \        # open(TESTING_DATA10, 'w') as testing_data10, \        # open(TESTING_DATA11, 'w') as testing_data11, \        # open(TESTING_DATA12, 'w') as testing_data12, \        # open(TESTING_DATA13, 'w') as testing_data13, \        # open(TESTING_DATA14, 'w') as testing_data14, \        # open(TESTING_DATA15, 'w') as testing_data15, \        # open(TESTING_DATA16, 'w') as testing_data16, \        # open(TESTING_DATA17, 'w') as testing_data17, \        # open(TESTING_DATA18, 'w') as testing_data18, \        # open(TESTING_DATA19, 'w') as testing_data19, \        # open(TESTING_DATA20, 'w') as testing_data20, \        # open(TESTING_DATA21, 'w') as testing_data21, \        # open(TESTING_DATA22, 'w') as testing_data22, \        # open(TESTING_DATA23, 'w') as testing_data23, \        mode_0 = 0        mode_1 = 0        mode_2 = 0        mode_3 = 0        mode_4 = 0        mode_5 = 0        mode_6 = 0        mode_7 = 0        mode_8 = 0        mode_9 = 0        mode_10 = 0        mode_11 = 0        mode_12 = 0        mode_13 = 0        mode_14 = 0        mode_15 = 0        mode_16 = 0        mode_17 = 0        mode_18 = 0        mode_19 = 0        mode_20 = 0        mode_21 = 0        mode_22 = 0        mode_23 = 0        mode_24 = 0        mode_25 = 0        mode_26 = 0        mode_27 = 0        mode_28 = 0        mode_29 = 0        mode_30 = 0        mode_31 = 0        mode_32 = 0        mode_33 = 0        mode_34 = 0        mode_37 = 0        mode_38 = 0        for line in orig_data:            if line[-3:-2] == ',':                # print("yes, it is a comma.")                last_char_in_line = int(line[-2:-1])            else:                last_char_in_line = int(line[-3:-1])            mode = last_char_in_line            if mode == 0:                mode_0 += 1                if mode_0 <= x_ordered_dict['mode_0']-100:                    training_data.write(line)                elif x_ordered_dict['mode_0']-100 < mode_0 <= x_ordered_dict['mode_0']:                    testing_data.write(line)                    testing_data0.write(line)            elif mode == 1:                mode_1 += 1                if mode_1 <= x_ordered_dict['mode_1'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_1'] - 100 < mode_1 <= x_ordered_dict['mode_1']:                    testing_data.write(line)                    testing_data1.write(line)            elif mode == 2:                mode_2 += 1                if mode_2 <= x_ordered_dict['mode_2'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_2'] - 100 < mode_2 <= x_ordered_dict['mode_2']:                    testing_data.write(line)                    testing_data2.write(line)            elif mode == 3:                mode_3 += 1                if mode_3 <= x_ordered_dict['mode_3'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_3'] - 100 < mode_3 <= x_ordered_dict['mode_3']:                    testing_data.write(line)                    # testing_data3.write(line)            elif mode == 4:                mode_4 += 1                if mode_4 <= x_ordered_dict['mode_4'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_4'] - 100 < mode_4 <= x_ordered_dict['mode_4']:                    testing_data.write(line)                    # testing_data4.write(line)            elif mode == 5:                mode_5 += 1                if mode_5 <= x_ordered_dict['mode_5'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_5'] - 100 < mode_5 <= x_ordered_dict['mode_5']:                    testing_data.write(line)                    # testing_data5.write(line)            elif mode == 6:                mode_6 += 1                if mode_6 <= x_ordered_dict['mode_6'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_6'] - 100 < mode_6 <= x_ordered_dict['mode_6']:                    testing_data.write(line)                    # testing_data6.write(line)            elif mode == 7:                mode_7 += 1                if mode_7 <= x_ordered_dict['mode_7'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_7'] - 100 < mode_7 <= x_ordered_dict['mode_7']:                    testing_data.write(line)                    # testing_data7.write(line)            elif mode == 8:                mode_8 += 1                if mode_8 <= x_ordered_dict['mode_8'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_8'] - 100 < mode_8 <= x_ordered_dict['mode_8']:                    testing_data.write(line)                    # testing_data8.write(line)            elif mode == 9:                mode_9 += 1                if mode_9 <= x_ordered_dict['mode_9'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_9'] - 100 < mode_9 <= x_ordered_dict['mode_9']:                    testing_data.write(line)                    # testing_data9.write(line)            elif mode == 10:                mode_10 += 1                if mode_10 <= x_ordered_dict['mode_10'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_10'] - 100 < mode_10 <= x_ordered_dict['mode_10']:                    testing_data.write(line)                    # testing_data10.write(line)            elif mode == 11:                mode_11 += 1                if mode_11 <= x_ordered_dict['mode_11'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_11'] - 100 < mode_11 <= x_ordered_dict['mode_11']:                    testing_data.write(line)                    # testing_data11.write(line)            elif mode == 12:                mode_12 += 1                if mode_12 <= x_ordered_dict['mode_12'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_12'] - 100 < mode_12 <= x_ordered_dict['mode_12']:                    testing_data.write(line)                    # testing_data12.write(line)            elif mode == 13:                mode_13 += 1                if mode_13 <= x_ordered_dict['mode_13'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_13'] - 100 < mode_13 <= x_ordered_dict['mode_13']:                    testing_data.write(line)                    # testing_data13.write(line)            elif mode == 14:                mode_14 += 1                if mode_14 <= x_ordered_dict['mode_14'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_14'] - 100 < mode_14 <= x_ordered_dict['mode_14']:                    testing_data.write(line)                    # testing_data14.write(line)            elif mode == 15:                mode_15 += 1                if mode_15 <= x_ordered_dict['mode_15'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_15'] - 100 < mode_15 <= x_ordered_dict['mode_15']:                    testing_data.write(line)                    # testing_data15.write(line)            elif mode == 16:                mode_16 += 1                if mode_16 <= x_ordered_dict['mode_16'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_16'] - 100 < mode_16 <= x_ordered_dict['mode_16']:                    testing_data.write(line)                    # testing_data16.write(line)            elif mode == 17:                mode_17 += 1                if mode_17 <= x_ordered_dict['mode_17'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_17'] - 100 < mode_17 <= x_ordered_dict['mode_17']:                    testing_data.write(line)                    # testing_data17.write(line)            elif mode == 18:                mode_18 += 1                if mode_18 <= x_ordered_dict['mode_18'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_18'] - 100 < mode_18 <= x_ordered_dict['mode_18']:                    testing_data.write(line)                    # testing_data18.write(line)            elif mode == 19:                mode_19 += 1                if mode_19 <= x_ordered_dict['mode_19'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_19'] - 100 < mode_19 <= x_ordered_dict['mode_19']:                    testing_data.write(line)                    # testing_data19.write(line)            elif mode == 20:                mode_20 += 1                if mode_20 <= x_ordered_dict['mode_20'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_20'] - 100 < mode_20 <= x_ordered_dict['mode_20']:                    testing_data.write(line)                    # testing_data20.write(line)            elif mode == 21:                mode_21 += 1                if mode_21 <= x_ordered_dict['mode_21'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_21'] - 100 < mode_21 <= x_ordered_dict['mode_21']:                    testing_data.write(line)                    # testing_data21.write(line)            elif mode == 22:                mode_22 += 1                if mode_22 <= x_ordered_dict['mode_22'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_22'] - 100 < mode_22 <= x_ordered_dict['mode_22']:                    testing_data.write(line)                    # testing_data22.write(line)            elif mode == 23:                mode_23 += 1                if mode_23 <= x_ordered_dict['mode_23'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_23'] - 100 < mode_23 <= x_ordered_dict['mode_23']:                    testing_data.write(line)                    # testing_data23.write(line)            elif mode == 24:                mode_24 += 1                if mode_24 <= x_ordered_dict['mode_24'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_24'] - 100 < mode_24 <= x_ordered_dict['mode_24']:                    testing_data.write(line)                    testing_data24.write(line)            elif mode == 25:                mode_25 += 1                if mode_25 <= x_ordered_dict['mode_25'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_25'] - 100 < mode_25 <= x_ordered_dict['mode_25']:                    testing_data.write(line)                    testing_data25.write(line)            elif mode == 26:                mode_26 += 1                if mode_26 <= x_ordered_dict['mode_26'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_26'] - 100 < mode_26 <= x_ordered_dict['mode_26']:                    testing_data.write(line)                    testing_data26.write(line)            elif mode == 27:                mode_27 += 1                if mode_27 <= x_ordered_dict['mode_27'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_27'] - 100 < mode_27 <= x_ordered_dict['mode_27']:                    testing_data.write(line)                    testing_data27.write(line)            elif mode == 28:                mode_28 += 1                if mode_28 <= x_ordered_dict['mode_28'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_28'] - 100 < mode_28 <= x_ordered_dict['mode_28']:                    testing_data.write(line)                    testing_data28.write(line)            elif mode == 29:                mode_29 += 1                if mode_29 <= x_ordered_dict['mode_29'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_29'] - 100 < mode_29 <= x_ordered_dict['mode_29']:                    testing_data.write(line)                    testing_data29.write(line)            elif mode == 30:                mode_30 += 1                if mode_30 <= x_ordered_dict['mode_30'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_30'] - 100 < mode_30 <= x_ordered_dict['mode_30']:                    testing_data.write(line)                    testing_data30.write(line)            elif mode == 31:                mode_31 += 1                if mode_31 <= x_ordered_dict['mode_31'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_31'] - 100 < mode_31 <= x_ordered_dict['mode_31']:                    testing_data.write(line)                    testing_data31.write(line)            elif mode == 32:                mode_32 += 1                if mode_32 <= x_ordered_dict['mode_32'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_32'] - 100 < mode_32 <= x_ordered_dict['mode_32']:                    testing_data.write(line)                    testing_data32.write(line)            elif mode == 33:                mode_33 += 1                if mode_33 <= x_ordered_dict['mode_33'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_33'] - 100 < mode_33 <= x_ordered_dict['mode_33']:                    testing_data.write(line)                    testing_data33.write(line)            elif mode == 34:                mode_34 += 1                if mode_34 <= x_ordered_dict['mode_34'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_34'] - 100 < mode_34 <= x_ordered_dict['mode_34']:                    testing_data.write(line)                    testing_data34.write(line)            elif mode == 37:                mode_37 += 1                if mode_37 <= x_ordered_dict['mode_37'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_37'] - 100 < mode_37 <= x_ordered_dict['mode_37']:                    testing_data.write(line)                    testing_data35.write(line)            elif mode == 38:                mode_38 += 1                if mode_38 <= x_ordered_dict['mode_38'] - 100:                    training_data.write(line)                elif x_ordered_dict['mode_38'] - 100 < mode_38 <= x_ordered_dict['mode_38']:                    testing_data.write(line)                    testing_data36.write(line)x_ordered_dict = csv_mode_preprocessing(OUTPUT_FILE=ORIG_DATA)data_generator(ORIG_DATA=ORIG_DATA, TRAINING_DATA=TRAINING_DATA, TESTING_DATA=TESTING_DATA,               x_ordered_dict=x_ordered_dict,               TESTING_DATA0=TESTING_DATA0,               TESTING_DATA1=TESTING_DATA1,               TESTING_DATA2=TESTING_DATA2,               # TESTING_DATA3=TESTING_DATA3,               # TESTING_DATA4=TESTING_DATA4,               # TESTING_DATA5=TESTING_DATA5,               # TESTING_DATA6=TESTING_DATA6,               # TESTING_DATA7=TESTING_DATA7,               # TESTING_DATA8=TESTING_DATA8,               # TESTING_DATA9=TESTING_DATA9,               # TESTING_DATA10=TESTING_DATA10,               # TESTING_DATA11=TESTING_DATA11,               # TESTING_DATA12=TESTING_DATA12,               # TESTING_DATA13=TESTING_DATA13,               # TESTING_DATA14=TESTING_DATA14,               # TESTING_DATA15=TESTING_DATA15,               # TESTING_DATA16=TESTING_DATA16,               # TESTING_DATA17=TESTING_DATA17,               # TESTING_DATA18=TESTING_DATA18,               # TESTING_DATA19=TESTING_DATA19,               # TESTING_DATA20=TESTING_DATA20,               # TESTING_DATA21=TESTING_DATA21,               # TESTING_DATA22=TESTING_DATA22,               # TESTING_DATA23=TESTING_DATA23,               TESTING_DATA24=TESTING_DATA24,               TESTING_DATA25=TESTING_DATA25,               TESTING_DATA26=TESTING_DATA26,               TESTING_DATA27=TESTING_DATA27,               TESTING_DATA28=TESTING_DATA28,               TESTING_DATA29=TESTING_DATA29,               TESTING_DATA30=TESTING_DATA30,               TESTING_DATA31=TESTING_DATA31,               TESTING_DATA32=TESTING_DATA32,               TESTING_DATA33=TESTING_DATA33,               TESTING_DATA34=TESTING_DATA34,               TESTING_DATA35=TESTING_DATA35,               TESTING_DATA36=TESTING_DATA36,               )