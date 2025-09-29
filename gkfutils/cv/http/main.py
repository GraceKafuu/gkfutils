import os
import requests



def main():
    url = "http://10.10.11.150:9996/belt/v1/getBeltStatus?deviceId=1甲"
    response = requests.get(url)
    response = response.content.decode('utf-8')
    print(response, type(response))
    print(eval(response), type(eval(response)))
    print(eval(response)["data"])

    status = eval(response)["data"]["deviceStatus"]
    print(status, type(status))


if __name__ == '__main__':
    main()