from neurons.validator.get_synapse import get_synapse

def test_get_synapse():
    result = get_synapse()
    assert result is not None
    
    print(result)
