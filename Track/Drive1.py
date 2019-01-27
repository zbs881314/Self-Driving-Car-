import socketio
import eventlet
from flask import Flask


sio = socketio.Server()

app = Flask(__name__)    # main

@sio.on('connect')
def connect(sid, environ):
    print('connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)