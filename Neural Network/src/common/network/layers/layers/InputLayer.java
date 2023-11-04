package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

public class InputLayer extends Layer{

	public InputLayer(int inputs) {
		super(inputs, inputs);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		this.lastActivation = input;
		return input;
	}

	@Override
	public void backprop() {}
	
	@Override
	public String name() {
		return "Input";
	}
}
