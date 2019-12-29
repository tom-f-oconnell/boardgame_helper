#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time
from datetime import datetime

import cv2
from aiohttp import web
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription


ROOT = os.path.dirname(__file__)
data_dir = os.path.join(ROOT, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

logger = logging.getLogger('pc')
pcs = set()


class VideoTrack(MediaStreamTrack):
    """
    A video stream track for processing frames.
    """
    kind = 'video'

    def __init__(self, track, transform, save_interval_s=60, ip=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.save_interval_s = save_interval_s
        self.last_save_time = None
        self.client_ip = ip

    # TODO maybe use frame.key_frame to decide when to process or save
    # something (in addition to timeout?)?

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == 'edges':
            # perform edge detection
            img = frame.to_ndarray(format='bgr24')
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            # TODO TODO TODO maybe should return immediately here and then save
            # / process elsewhere, for async purposes?
            #return new_frame
            ret = new_frame

        elif self.transform == 'rotate':
            # rotate image
            img = frame.to_ndarray(format='bgr24')
            rows, cols, _ = img.shape
            # TODO what is the frame.time here for? actually time?
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45,
                1
            )
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            #return new_frame
            ret = new_frame

        else:
            ret = frame
            #return frame

        if (self.last_save_time is None or
            time.time() - self.last_save_time >= self.save_interval_s):

            if self.client_ip is None:
                fname = ''
            else:
                fname = self.client_ip.replace('.','-') + '_'
            fname += datetime.strftime(datetime.now(), '%y%m%d_%H%M%S') + '.jpg'
            fname = os.path.join(data_dir, fname)

            # TODO are first / key (not saying they are necessarily synonymous)
            # frames of lower quality often? should i only save starting on
            # second / next non-key frame or something?

            img = frame.to_ndarray(format='bgr24')
            #
            print(f'writing frame to {fname}')
            #
            cv2.imwrite(fname, img)
            self.last_save_time = time.time()

        return ret


async def index(request):
    content = open(os.path.join(ROOT, 'index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, 'client.js'), 'r').read()
    return web.Response(content_type='application/javascript', text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    pc_id = 'PeerConnection(%s)' % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + ' ' + msg, *args)

    log_info('Created for %s', request.remote)
    
    # TODO should i use something more unidirectional since move recommedations
    # should only go one way? or use this for all kinds of data, including stuff
    # selected in client, etc? other more basic HTTP idioms for this stuff?
    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            if isinstance(message, str) and message.startswith('ping'):
                channel.send('pong' + message[4:])

    # TODO possible to do away with this?
    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        log_info('ICE connection state is %s', pc.iceConnectionState)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        log_info('Track %s received', track.kind)

        assert track.kind == 'video'
        local_video = VideoTrack(track, transform=params['video_transform'],
            ip=request.remote
        )
        pc.addTrack(local_video)

        @track.on('ended')
        async def on_ended():
            log_info('Track %s ended', track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps(
            {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WebRTC audio / video / data-channels demo'
    )
    parser.add_argument('--cert-file', help='SSL certificate file (for HTTPS)')
    parser.add_argument('--key-file', help='SSL key file (for HTTPS)')
    parser.add_argument(
        '--port', type=int, default=8080, help='Port for HTTP server (default: 8080)'
    )
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)

