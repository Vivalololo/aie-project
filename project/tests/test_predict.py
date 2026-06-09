from src.predict import predict_image


def test_predict_returns_dict():

    result = predict_image("./data/test_image.jpeg")

    assert isinstance(result, dict)

    assert "class" in result
    assert "confidence" in result