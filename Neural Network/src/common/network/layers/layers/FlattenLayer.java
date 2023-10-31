package common.network.layers.layers;

public class FlattenLayer extends Layer {

	public FlattenLayer(int inputs, int inDepth) {
		super(inputs, inputs*inDepth);
		lastActivation = new float[outputs][1];
		depth = 1;
	}

	public FlattenLayer(Layer last, Layer next) {
		super(last, next);
		lastActivation = new float[outputs][1];
		depth = 1;
	}

	public FlattenLayer(Layer last) {
		super(last, last.outputs * last.depth);
		lastActivation = new float[outputs][1];
		depth = 1;
	}

	@Override
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		int index = 0;
		for(int i = 0; i < input.length; i++)
		{
			for(int j = 0; j < input[0].length; j++)
			{
				lastActivation[index][0] = input[i][j];
				index++;
			}
		}
		return lastActivation;
	}

	@Override
	public void backprop() {
		float[][] nextErrorWeighted = getGradient();	
		clearGradients();
		
		int index = 0;
		
		float[][] out = new float[lastLayer.outputs][lastLayer.depth];
		
		for(int i = 0; i < lastLayer.outputs; i++)
		{
			for(int j = 0; j < lastLayer.depth; j++)
			{
				out[i][j] = nextErrorWeighted[index][0];
				index++;
			}
		}
		lastLayer.reportGradient(out);
	}

	@Override
	public String name() {
		return "Flatten";
	}

}
