package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * A layer that transposes its input.
 * @author C. Cooper
 */
public class RotationLayer extends Layer{

	/**
	 * Creates a RotationLayer which transposes the given layer.
	 * @param last The layer to transpose.
	 */
	public RotationLayer(Layer last) {
		super(last, last.depth);
		depth = last.outputs;
		lastActivation = new SimpleMatrix(outputs, depth);
		setGradientSize(outputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		masks = lastLayer.getMasks();//Pulls the last layer's masks forward.
		lastActivation = lastLayer.getLastActivation().transpose();
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		lastLayer.reportGradient(nextErrorWeighted.transpose());//Just transposes the gradient and passes it back.
	}

	@Override
	public String name() {
		return "Rotate";
	}
	
	@Override
	public String stringify() {
		/*
		 * Returns a string in the form:
		 * thisId lastLayerId numOutputs depth
		 */
		return getId() + " " + lastLayer.getId() + " " + outputs + " " + depth + "\n"; 
	}
	
	/**
	 * Loads a RotationLayer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static RotationLayer load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();//Gets this layer's id
		int lastID = scanner.nextInt();//Gets the last layer's id
		scanner.close();
		RotationLayer out = new RotationLayer(model.getLayerByID(lastID));//Makes the RotationLayer
		out.setId(id);//Sets the RotationLayer's id (so that future layers can use model.getLayerByID(id) to find it).
		return out;
	}
	
	@Override
	public String className() {
		return "Rotation";
	}
}
