package common.network.layers.layers;

import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

public class StandardLayer extends Layer{

	Activation activation;
	double[][][] weights;
	double[][] biases;
	
	private SimpleMatrix weightedInputs;
	
	public StandardLayer(int inputs, int outputs, Activation activation) {
		super(inputs, outputs);
		biases = new double[outputs][depth];
		weights = new double[outputs][depth][inputs];
		this.activation = activation;
		initHe();
	}
	
	public StandardLayer(Layer inputLayer, int outputs, Activation activation) {
		super(inputLayer, outputs);
		depth = inputLayer.depth;
		biases = new double[outputs][depth];
		weights = new double[outputs][depth][inputs];
		this.activation = activation;
		initHe();
	}
	
	public void init()
	{
		biases = SimpleMatrix.random(outputs, depth).toArray2();
		for(int i = 0; i < outputs; i++)
			weights[i] = SimpleMatrix.random(depth, inputs).toArray2();
	}
	
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		masks = lastLayer.getMasks();
		double[][] in = lastLayer.getLastActivation().toArray2();
		double[][] weightedInputs = new double[outputs][depth];
		for(int o = 0; o < outputs; o++)
		{
			for(int d = 0; d < depth; d++)
			{
				weightedInputs[o][d] = biases[o][d];
				for(int i = 0; i < inputs; i++)
				{
					weightedInputs[o][d] += weights[o][d][i] * in[i][d];
				}
			}
		}
		this.weightedInputs = new SimpleMatrix(weightedInputs);
		lastActivation = activation.activation(this.weightedInputs);
		return lastActivation;
	}
	
	@Override
	public void backprop()
	{
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		double[][] error = activation.error(weightedInputs, nextErrorWeighted).toArray2();

		double[][] thisErrorWeighted = new double[inputs][depth];
		for(int d = 0; d < depth; d++)
		{
			for(int o = 0; o < outputs; o++)
			{
				double thisError = error[o][d] * model.getLearningRate();
				biases[o][d] -= thisError;
				for(int i = 0; i < inputs; i++)
				{
					thisErrorWeighted[i][d] += weights[o][d][i] * error[o][d];
					weights[o][d][i] -= lastLayer.getLastActivation().get(i, d) * thisError;
				}
			}
		}
		
		lastLayer.reportGradient(new SimpleMatrix(thisErrorWeighted));
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

        weights = new double[outputs][depth][inputs];
        
    	for(int j = 0; j < outputs; j++)
    		for(int i = 0; i < depth; i++)
    			for(int k = 0; k < inputs; k++)
    				weights[j][i][k] = mean + desiredStdDev * random.nextGaussian();	
	}
	
	@Override
	public String className() {
		return "Standard";
	}
	
	@Override
	public String stringify() {
		// TODO Auto-generated method stub
		return "";
	}
	
	public static StandardLayer load(String e, LayersModel f, int g) {return null;}
	
	/*@Override
	public String stringify()
	{
		String out = getId() + " " + lastLayer.getId() + " " + inputs  + " " + outputs + " " + depth + " " + activation.name() +"\n";
		for(int d = 0; d < depth; d++)
		{
			for(int o = 0; o < outputs; o++)
			{
				out += biases.get(o, d) + " ";
				for(int i = 0; i < inputs; i++)
				{
					out += weights[d].get(o, i) + " ";
				}
			}
			out += "\n";
		}
		return out;
	}

	public static StandardLayer2 load(String string, LayersModel model, int position) {
		StandardLayer2 out;

		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		int inputs = scanner.nextInt();
		int outputs = scanner.nextInt();
		int depth = scanner.nextInt();
		String activation = scanner.next();
		Activation activation2 = getActivation(activation);
		out = new StandardLayer2(model.getLayerByID(lastID), outputs, activation2);
		SimpleMatrix biases = new SimpleMatrix(outputs, depth);
		SimpleMatrix[] weights = new SimpleMatrix[depth];
		
		for(int d = 0; d < depth; d++)
		{
			weights[d] = new SimpleMatrix(outputs, inputs);
			for(int o = 0; o < outputs; o++)
			{
				biases.set(o, d, scanner.nextDouble());
				for(int i = 0; i < inputs; i++)
				{
					weights[d].set(o, i, scanner.nextDouble());
				}
			}
		}
		out.biases = biases.toArray2();
		out.weights = weights.;
		
		scanner.close();
		
		out.setId(id);
		
		return out;
	}*/
	
	static Activation getActivation(String name)
	{
		switch(name)
		{
		case "Softmax":
			return Activation.SOFTMAX;
		case "Softmax_Depthwise":
			return Activation.SOFTMAX_DEPTHWISE;
		case "ReLU":
			return Activation.RELU;
		case "Sigmoid":
			return Activation.SIGMOID;
		case "None":
			return Activation.NONE;
		default:
			throw new IllegalArgumentException("Unknown Activation: " + name);
		}
	}
}
