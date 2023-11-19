package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * A layer that flattens the last layer's outputs to a n x 1 matrix, where n is the last layer's output
 * count * its depth.
 * <br><br>
 * Note: This layer hasn't been tested since moving to EJML, so there is a real chance that it doesn't 
 * work correctly, because I don't know how the {@link SimpleMatrix#reshape(int, int)} method works exactly.
 * If the way it works in consistent with common reshaping convention, this should work perfectly.
 * @author C. Cooper
 */
public class FlattenLayer extends Layer {

	public FlattenLayer(Layer last) {
		super(last, last.outputs * last.depth);
		lastActivation = new SimpleMatrix(outputs, 1);
		depth = 1;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		lastActivation = lastLayer.getLastActivation().copy();
		lastActivation.reshape(outputs, 1);//Flattens the activation
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();	
		clearGradients();
		
		nextErrorWeighted.reshape(inputs, depth);//Unflattens the gradient (this is the part that might not work)
		lastLayer.reportGradient(nextErrorWeighted);
	}

	@Override
	public String name() {
		return "Flatten";
	}

	@Override
	public String stringify() {
		/*
		 * Returns a String in the form:
		 * thisId lastLayerId
		 */
		return getId() + " " + lastLayer.getId();
	}
	
	/**
	 * Loads a FlattenLayer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @return An AttentionLayer based on the given String.
	 */
	public static FlattenLayer load(String string, LayersModel model)
	{
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastId = scanner.nextInt();
		scanner.close();
		FlattenLayer out = new FlattenLayer(model.getLayerByID(lastId));
		out.setId(id);
		return out;
	}
}
