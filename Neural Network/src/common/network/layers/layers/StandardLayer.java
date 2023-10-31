package common.network.layers.layers;

import java.util.Random;

import common.network.layers.Activation;
import common.network.math.NetworkMath;

public class StandardLayer extends Layer{

	Activation activation;
	float[][][] weights;
	float[][] biases;
	
	private float[][] weightedInputs;
	
	public StandardLayer(int inputs, int outputs, Activation activation) {
		super(inputs, outputs);
		biases = new float[outputs][depth];
		weights = new float[outputs][depth][inputs];
		this.activation = activation;
		initHe();
	}
	
	public StandardLayer(Layer inputLayer, int outputs, Activation activation) {
		super(inputLayer, outputs);
		depth = inputLayer.depth;
		biases = new float[outputs][depth];
		weights = new float[outputs][depth][inputs];
		this.activation = activation;
		initHe();
	}
	
	public void init()
	{
		Random random = new Random();
		for(int d = 0; d < depth; d++)
		{
			for(int i = 0; i < outputs; i++)
			{
				biases[i][d] = random.nextFloat();
				if(biases[i][d] == 0)biases[i][d] = 1;
				for(int j = 0; j <  inputs; j++)
				{
					weights[i][d][j] = random.nextFloat();
					if(weights[i][d][j] == 0)weights[i][d][j] = 1;
				}
			}
		}
	}

	@Override
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		
		weightedInputs = new float[outputs][depth];
		for(int d = 0; d < depth; d++)
		{
			for(int i = 0; i < outputs; i++)
			{
				for(int j = 0; j < inputs; j++)
				{
					weightedInputs[i][d] += weights[i][d][j] * input[j][d];
				}
				weightedInputs[i][d] += biases[i][d];
			}
		}
		lastActivation = activation.activation(weightedInputs);
		return lastActivation;
	}
	
	@Override
	public void backprop()
	{
		float[][] nextErrorWeighted = getGradient();
		clearGradients();
		
		float[][] error = activation.error(weightedInputs, nextErrorWeighted);
		//System.out.println(LayersMain.arrayToString(error));
		float[][] thisErrorWeighted = new float[inputs][depth];
		for(int d = 0; d < depth; d++)
		{
			for(int i = 0; i < outputs; i++)
			{
				for(int j = 0; j < inputs; j++)
				{
					thisErrorWeighted[j][d]  += weights[i][d][j] * error[i][d];
					weights[i][d][j] -= lastLayer.getLastActivation()[j][d] * error[i][d] * model.getLearningRate();
				}
				biases[i][d] -= error[i][d] * model.getLearningRate();
			}
		}
		
		lastLayer.reportGradient(thisErrorWeighted);
	}
	
	@Override
	public String name() {
		return "Standard";
	}
	
	@Override
	public String toString() {
		//String out = outputs + " x " + inputs + "\n";
		//out += LayersMain.floatMatrixToString(weights, 2);
		//return out;
		
		return "Standard [" + activation.name() + "] (" + outputs + ", " + depth + ")";
	}
	
	public void initHe()
	{
		double desiredVariance = 2d / inputs;
        double desiredStdDev = Math.sqrt(desiredVariance);

        Random random = new Random();
        double mean = 0f;

        for(int i = 0; i < outputs; i++)
        	for(int j = 0; j < depth; j++)
        		for(int k = 0; k < inputs; k++)
        			weights[i][j][k] = (float) (mean + desiredStdDev * random.nextGaussian());
	}
}
