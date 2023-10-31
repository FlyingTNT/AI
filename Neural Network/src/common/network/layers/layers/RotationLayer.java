package common.network.layers.layers;

public class RotationLayer extends Layer{

	public RotationLayer(Layer last) {
		super(last, last.depth);
		depth = last.outputs;
		lastActivation = new float[outputs][depth];
	}

	@Override
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				lastActivation[i][j] = input[j][i];
			}
		}
		return lastActivation;
	}

	@Override
	public void backprop() {
		float[][] nextErrorWeighted = getGradient();	
		clearGradients();
		
		float[][] out = new float[depth][outputs];
		
		for(int i = 0; i < outputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				out[j][i] = nextErrorWeighted[i][j];
			}
		}
		lastLayer.reportGradient(out);
	}

	@Override
	public String name() {
		return "Rotate";
	}

}
