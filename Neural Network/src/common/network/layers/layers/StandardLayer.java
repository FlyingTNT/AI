package common.network.layers.layers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;

public class StandardLayer extends Layer{

	Activation activation;
	SimpleMatrix[] weights;//[depth]xOutxIn
	SimpleMatrix biases;//Out x Depth
	
	private SimpleMatrix weightedInputs;
	
	public StandardLayer(int inputs, int outputs, Activation activation) {
		super(inputs, outputs);
		biases = new SimpleMatrix(outputs, depth);
		weights = new SimpleMatrix[depth];
		this.activation = activation;
		initHe();
	}
	
	public StandardLayer(Layer inputLayer, int outputs, Activation activation) {
		super(inputLayer, outputs);
		depth = inputLayer.depth;
		biases = new SimpleMatrix(outputs, depth);
		weights = new SimpleMatrix[depth];
		this.activation = activation;
		initHe();
	}
	
	public void init()
	{
		biases = SimpleMatrix.random(outputs, depth);
		for(int i = 0; i < depth; i++)
			weights[i] = SimpleMatrix.random(outputs, inputs);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		masks = lastLayer.getMasks();
		input = lastLayer.getLastActivation();//Input x depth
		
		weightedInputs = new SimpleMatrix(outputs, depth);
		
		for(int i = 0; i < depth; i++)
			weightedInputs.setColumn(i, weights[i].mult(input.getColumn(i)));
		
		weightedInputs = weightedInputs.plus(biases);
			
		lastActivation = activation.activation(weightedInputs);
		return lastActivation;
	}
	
	public void backprop1()
	{
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		SimpleMatrix error = activation.error(weightedInputs, nextErrorWeighted).scale(model.getLearningRate());
		
		//error: output x depth
		//lastLayer.lastActivation: input x depth
		//weights: [depth] x outputs x inputs
		
		SimpleMatrix thisErrorWeighted = new SimpleMatrix(inputs, depth);
		
		for(int i = 0; i < depth; i++)
		{
			thisErrorWeighted.setColumn(i, weights[i].transpose().mult(error.getColumn(i)));
			weights[i] = weights[i].minus(error.getColumn(i).mult(lastLayer.getLastActivation().getColumn(i).transpose()));
		}
		
		biases = biases.minus(error);
		
		double norm = thisErrorWeighted.normF();
		if(norm > 1)
			thisErrorWeighted = thisErrorWeighted.scale(1/norm);
		
		lastLayer.reportGradient(thisErrorWeighted);
	}
	
	@Override
	public void backprop()
	{
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		SimpleMatrix error1 = activation.error(weightedInputs, nextErrorWeighted);
		SimpleMatrix error = error1.scale(model.getLearningRate());
		//System.out.println(LayersMain.arrayToString(error));
		SimpleMatrix thisErrorWeighted = new SimpleMatrix(inputs, depth);
		for(int d = 0; d < depth; d++)
		{
			for(int i = 0; i < outputs; i++)
			{
				for(int j = 0; j < inputs; j++)
				{
					thisErrorWeighted.set(j, d, thisErrorWeighted.get(j, d) + weights[d].get(i, j) * error1.get(i, d));
					weights[d].set(i, j, weights[d].get(i, j) - lastLayer.getLastActivation().get(j, d) * error.get(i, d));
				}
			}
		}
		
		biases.minus(error);
		
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

        	for(int j = 0; j < depth; j++)
        	{
        		weights[j] = new SimpleMatrix(outputs, inputs); 
        		for(int i = 0; i < outputs; i++)
        		for(int k = 0; k < inputs; k++)
        			weights[j].set(i, k, (mean + desiredStdDev * random.nextGaussian()));
        	}
        		
	}
}
