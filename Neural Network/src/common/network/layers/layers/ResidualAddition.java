package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * Residual addition layer. Adds the activations of last and residual layers together.
 * Used to combat vanishing gradients by giving them a path around layers that might cause vanishing gradients.
 * Ex:<br>
 * NormLayer -> AttentionLayer -> ResidualAddition<br><pre>
 *      ↘----------------------↗</pre><br>
 * The attention layer tends to cause vanishing gradients, so the ResidualAddition bypasses it.
 * @author C. Cooper
 */
public class ResidualAddition extends Layer {

	private final Layer residual;//The layer the residual connection is to
	
	/**
	 * Creates a ResidualAddition layer that connects to the given layers.
	 * <br><br>
	 * Which one is last and which is residual doesn't actually have any effect, although last is generally the layer being
	 * bypassed.
	 * @param last The layer being bypassed.
	 * @param residual The second layer to connect to (generally before the layer being bypassed).
	 */
	public ResidualAddition(Layer last, Layer residual) {
		super(last, last.outputs);
		this.residual = residual;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		masks = lastLayer.getMasks();//Pulls the last layer's masks forward
		lastActivation = lastLayer.getLastActivation().plus(residual.getLastActivation());//Adds the activations of the last and residual layers.
		return lastActivation;
	}

	/**
	 * Performs backpropogation. Just passes the gradients given to this layer back to the last and residual layers.
	 */
	@Override
	public void backprop() {
		SimpleMatrix gradient = getGradient();//Gets the gradient.
		clearGradients();//Clears the gradients so that this backprop's gradients don't affect the next.
		lastLayer.reportGradient(gradient);//Pass the gradient back to the last layer
		residual.reportGradient(gradient);//Pass the gradient back to the residual layer
	}

	@Override
	public String name() {
		return "Residual Addition";
	}
	
	@Override
	public String className() {
		return "ResidualAddition";
	}
	
	@Override
	public String stringify() {
		/*
		 * Returns a String with the form:
		 * thisId laastId residualId #inputs depth
		 */
		return getId() + " " + lastLayer.getId() + " " + residual.getId() + " " + inputs + " " + depth;
	}
	
	/**
	 * Loads a ResidualAddition based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static ResidualAddition load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();//Gets this layer's id
		int firstID = scanner.nextInt();//Gets the id of the last layer
		int secondID = scanner.nextInt();//Gets the id of the residual layer
		scanner.close();
		ResidualAddition out = new ResidualAddition(model.getLayerByID(firstID), model.getLayerByID(secondID));
		out.setId(id);
		return out;
	}
}
