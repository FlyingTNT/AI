package common.network.layers.layers;

import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

/**
 * A standard layer. (also called linear, dense, fully-connected, or feed forward). I'll probably
 * rename this to LinearLayer in the future.
 * @author C. Cooper
 */
public class StandardLayer extends Layer{

	Activation activation;//The activation function to apply
	float[][][][] weights;//The array of weights [outputs][depth][inputs][depth]
	float[][] biases;//The array of biases [outputs][depth]
	
	private SimpleMatrix weightedInputs;//The weighted inputs to the last activation. This is a SimpleMatrix b/c the activation functions take in SimpleMatricies
	
	/**
	 * Basic constructor. Gets its input and depth dimensions from the last layer's output and depth dimensions.
	 * @param inputLayer The layer this layer gets its inputs from.
	 * @param outputs The number of outputs this layer has.
	 * @param activation The activation function to apply to this layer.
	 */
	public StandardLayer(Layer inputLayer, int outputs, Activation activation) {
		super(inputLayer, outputs);
		depth = inputLayer.depth;
		biases = new float[outputs][depth];
		this.activation = activation;
		initHe();//Initializes the weights using He initialization. See the method doc for why.
	}
	
	/**
	 * Basic initialization function that just sets the weights and biases to random numbers 0-1
	 */
	private void init()
	{
		biases = toFloat(SimpleMatrix.random(outputs, depth).toArray2());
		weights = new float[outputs][depth][][];
		for(int i = 0; i < outputs; i++)
			for(int j = 0; j < depth; j++)
				weights[i][j] = toFloat(SimpleMatrix.random(inputs, depth).minus(0.5d).toArray2());
	}
	
	/**
	 * Standard linear layer activation function. Each output has a weight to each input and the value of the output is
	 * the sum of all of the input values times each of their weights.
	 * <br><br>
	 * Does not use EJML internally because I found that to be slower than the naive implementation.
	 * @param input The input to this layer (ignored. Just uses the last layer's {@link Layer#getLastActivation() getLastActivation()} function)
	 * @param isInference Whether or not the model is performing inference (no effect).
	 * @return This layer's activation.
	 */
	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		masks = lastLayer.getMasks();//Pulls the last layer's masks forward.
		
		/*
		 * This is just the naive linear layer activation. If you understand how that works, you should understand this.
		 */
		double[][] in = lastLayer.getLastActivation().toArray2();
		double[][] weightedInputs = new double[outputs][depth];
		for(int o = 0; o < outputs; o++)
		{
			for(int d = 0; d < depth; d++)
			{
				weightedInputs[o][d] = biases[o][d];
				for(int i = 0; i < inputs; i++)
				{
					for(int di = 0; di < depth; di++)
					{
						weightedInputs[o][d] += weights[o][d][i][di] * in[i][di];
					}
				}
			}
		}
		this.weightedInputs = new SimpleMatrix(weightedInputs);
		lastActivation = activation.activation(this.weightedInputs);
		//lastActivation.print();
		return lastActivation;
	}
	
	/**
	 * Performs backprop on this layer. The layer(s) that follow this layer should have reported their
	 * gradients to this layer using {@link #reportGradient(SimpleMatrix)}. It uses the same function to report
	 * its gradient to its preceding layer.
	 * <br><br>
	 * Does not use EJML internally because I found that to be slower.
	 */
	@Override
	public void backprop()
	{
		SimpleMatrix nextErrorWeighted = getGradient();//Gets the next gradient
		clearGradients();//Clears its internal record of the gradients so that this backprop's gradients don't affect the next
		
		double[][] error = activation.error(weightedInputs, nextErrorWeighted).toArray2();//Pulls the gradient through the activation function.

		double[][] thisErrorWeighted = new double[inputs][depth];//Matrix for the error of the inputs to this layer.
		for(int d = 0; d < depth; d++)
		{
			for(int o = 0; o < outputs; o++)
			{
				double thisError = error[o][d] * model.getLearningRate();//Scales the output error by the LR (so we don't do this same calculation for each input)
				biases[o][d] -= thisError;//Updates the biases
				for(int i = 0; i < inputs; i++)
				{
					for(int di = 0; di < depth; di++)
					{
						thisErrorWeighted[i][di] += weights[o][d][i][di] * error[o][d];//Pulls the error through the weights
						weights[o][d][i][di] -= lastLayer.getLastActivation().get(i, di) * thisError;//Updates the  weights
					}
				}
			}
		}
		
		lastLayer.reportGradient(new SimpleMatrix(thisErrorWeighted));//Pass the gradient down to the last layer.
	}
	
	@Override
	public String name() {
		return "Standard";
	}
	
	@Override
	public String toString() {		
		return "Standard [" + activation.name() + "] (" + outputs + ", " + depth + ")";
	}
	
	/**
	 * Uses He initialization to initialize the weights (just keeps the biases as zero). That is, inits the weights so that their mean is
	 * zero and their variance is 2/the number of inputs.
	 * <br><br>
	 * I use this over just random initialization, because with large input dims, random init tends to cause activations with very
	 * large magnitudes, which have the potential to cause overflows. 
	 * 
	 * @author C. Cooper
	 * @author ChatGPT
	 */
	private void initHe()
	{		
		//The code for generating a random number w/ given variance and mean comes from ChatGPT
		double desiredVariance = 2d / inputs;
        double desiredStdDev = Math.sqrt(desiredVariance);

        Random random = new Random();
        double mean = 0d;

        weights = new float[outputs][depth][inputs][depth];
        
    	for(int o = 0; o < outputs; o++)
    		for(int d = 0; d < depth; d++)
    			for(int i = 0; i < inputs; i++)
    				for(int di = 0; di < depth; di++)
    					weights[o][d][i][di] = (float) (mean + desiredStdDev * random.nextGaussian());	
	}
	
	@Override
	public String className() {
		return "Standard";
	}
	
	@Override
	public String stringify()
	{
		/*
		 * Creates a String with the form:
		 * thisId lastId inputCount outputCount depth activationName
		 * bias[0][0] weight[0][0][0] weight[0][0][1]...bias[0][1] weight[0][1][0] weight[0][1][1]...weight[0][depth-1][inputCount-1]
		 * bias[1][0] weight[1][0][0] weight[1][0][1]...bias[1][1] weight[1][1][0] weight[1][1][1]...weight[1][depth-1][inputCount-1]
		 */
		String out = getId() + " " + lastLayer.getId() + " " + inputs  + " " + outputs + " " + depth + " " + activation.name() +"\n";
		for(int o = 0; o < outputs; o++)
		{
			for(int d = 0; d < depth; d++)
			{
				out += biases[o][d] + " ";
				for(int i = 0; i < inputs; i++)
				{
					out += weights[o][d][i] + " ";
				}
			}
			out += "\n";
		}
		return out;
	}

	/**
	 * Loads a StandardLayer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static StandardLayer load(String string, LayersModel model, int position) {
		StandardLayer out;

		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		int inputs = scanner.nextInt();
		int outputs = scanner.nextInt();
		int depth = scanner.nextInt();
		String activation = scanner.next();
		Activation activation2 = getActivation(activation);
		out = new StandardLayer(model.getLayerByID(lastID), outputs, activation2);
		
		for(int o = 0; o < outputs; o++)
		{
			for(int d = 0; d < depth; d++)
			{
				out.biases[o][d] = scanner.nextFloat();
				for(int i = 0; i < inputs; i++)
				{
					//out.weights[o][d][i] = scanner.nextFloat();
				}
			}
		}
		
		scanner.close();
		
		out.setId(id);
		
		return out;
	}
	
	/**
	 * Given a String, returns the {@link Activation} with that name.
	 * @param name The name of the activation.
	 * @return The activation with the given name.
	 * @throws IllegalArgumentException If there is no Activation with that name.
	 */
	static Activation getActivation(String name)
	{
		switch(name){
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
	
	static float[][] toFloat(double[][] array)
	{
		float[][] out = new float[array.length][array[0].length];
		for(int i = 0; i < array.length; i++)
			for(int j = 0; j < array[0].length; j++)
				out[i][j] = (float)array[i][j];
		return out;
	}
}
