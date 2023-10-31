package common.network.layers.layers;

public class InputLayer extends Layer{

	public InputLayer(int inputs) {
		super(inputs, inputs);
	}

	@Override
	public float[][] activation(float[][] input) {
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
