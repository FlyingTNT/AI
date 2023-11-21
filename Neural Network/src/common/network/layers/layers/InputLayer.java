package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * Layer that takes in model inputs.
 * @author C. Cooper
 */
public class InputLayer extends Layer{

	public InputLayer(int inputs) {
		super(inputs, inputs);
	}
	
	public InputLayer(int inputs, int depth) {
		super(inputs, inputs);
		this.depth = depth;
		setGradientSize(inputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		this.lastActivation = input;
		return input;
	}

	@Override
	public void backprop() {return;}
	
	@Override
	public String name() {
		return "Input";
	}
	
	/**
	 * Does nothing because this layer doesn't learn.
	 * @param gradient The gradient to do nothing with.
	 */
	@Override
	public void reportGradient(SimpleMatrix gradient) {return;}
	
	@Override
	public String stringify() {
		return getId() + " " + inputs;//The only info this layer needs to rebuild itself are its id and input count
	}

	/**
	 * Creates an InputLayer from a String produced by this class's {@link #stringify() stringify()} method.
	 * <br><br>
	 * The string should have the format: "{id} {inputs}"
	 * @param string A string produced by {@link #stringify() stringify()}
	 * @param model The model this layer is in.
	 * @param position The position of this layer in the model (not used).
	 * @return An InputLayer described by the given string.
	 */
	public static InputLayer load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);//A scanner whose delimiter is whitespace.
		int id = scanner.nextInt();//Gets the id from the string
		int inputs = scanner.nextInt();//Gets the input count from the string
		scanner.close();
		
		InputLayer out = new InputLayer(inputs);
		out.setId(id);
		return out;
	}
}
