from wpilib import ADIS16448_IMU

# ADIS16448 plugged into the MXP port
self.gyro = ADIS16448_IMU()

# Function to read gyroscope data from the ADIS16448 IMU
def read_gyro_data():
    gyro_data = {
        'pitch': self.gyro.getPitch(),
        'roll': self.gyro.getRoll(),
        'yaw': self.gyro.getYaw()
    }
    return gyro_data


# this wrote isn't going to work, we need to use the ADIS16448_IMU hardware to get the gyro data