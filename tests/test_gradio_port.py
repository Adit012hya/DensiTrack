import socket

from app import resolve_gradio_port


def test_resolve_gradio_port_falls_back_when_default_busy():
    preferred_port = 7860

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", preferred_port))
        sock.listen(1)

        port = resolve_gradio_port(preferred_port=preferred_port, max_port=preferred_port + 5)

    assert port != preferred_port
    assert preferred_port < port <= preferred_port + 5
