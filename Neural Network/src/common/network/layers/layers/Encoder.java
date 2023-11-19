package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

/**
 * A transformer model's Encoder, as defined in Google's Attention is All You Need.
 * Consists of:<br>
 * An {@link AttentionLayer},<br>
 * A {@link ResidualAddition} layer that connects to the Attention layer, and the layer before the Encoder,<br>
 * A {@link NormLayer},<br>
 * A {@link StandardLayer} with ReLU activation,<br>
 * A {@link ResidualAddition} layer that connects to the StandardLayer and the NormLayer before it,<br>
 * And a {@link NormLayer}.
 * @author C. Cooper
 */
public class Encoder extends Layer{

	final AttentionLayer attention;//The Encoder's self attention layer
	final ResidualAddition attentionResidual;//The residual addition layer that bypasses the attention layer
	final NormLayer attentionNorm;//The attention layer's norm layer
	final StandardLayer linear;//The encoder's linear layer
	final ResidualAddition linearResidual;//The residual addition layer that bypasses the linear layer
	final NormLayer linearNorm;//The linear layer's layer norm layer.
	
	final Layer[] layers;//An array with all of the layers (so that I can do a for loop rather than listing every layer)

	/**
	 * Creates an Encoder whose inputs come from the given lastLayer, and whose attention layer has the given number of heads.
	 * @param lastLayer The layer this layer gets its inputs from.
	 * @param heads The number of heads in the AttentionLayer.
	 * @param masking Whether to do masking of padding tokens.
	 */
	public Encoder(Layer lastLayer, int heads, boolean masking) {
		super(lastLayer, lastLayer.outputs);//Super constructor with lastLayer = the given lastLayer and outputs = lastLayer.outputs
		this.depth = lastLayer.depth;
		
		attention = new AttentionLayer(lastLayer, lastLayer, lastLayer, heads, false, masking, false);//Creates a self-attention layer attending to the last layer
		attentionResidual = new ResidualAddition(attention, lastLayer);//Creates a ResidualAddition layer connecting to the attention and last layers
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, lastLayer.outputs, Activation.RELU);//Creates a linear layer connecting to the attentionNorm layer.
		linearResidual = new ResidualAddition(linear, attentionNorm);//Creates a ResidualAddition layer connecting to the linear and attentionNorm
		linearNorm = new NormLayer(linearResidual);
		
		layers = new Layer[] {attention, attentionResidual, attentionNorm, linear, linearResidual, linearNorm};
	}
	
	/**
	 * Constructor used for {@link #load(String, LayersModel, int)}.
	 * @param lastLayer The layer that precedes this layer.
	 * @param attention This layer's internal AttentionLayer
	 * @param linear This layer's internal StandardLayer
	 * @param residualAddition This layer's internal attention residual addition layer
	 * @param norm This layer's internal attention norm layer.
	 */
	private Encoder(Layer lastLayer, AttentionLayer attention, StandardLayer linear, ResidualAddition residualAddition, NormLayer norm) {
		super(lastLayer, lastLayer.outputs);
		this.depth = lastLayer.depth;
		
		this.attention = attention;
		attentionResidual = residualAddition;
		attentionNorm = norm;
		this.linear = linear;
		linearResidual = new ResidualAddition(linear, norm);
		linearNorm = new NormLayer(linearResidual);
		
		layers = new Layer[] {attention, residualAddition, norm, linear, linearResidual, linearNorm};
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		for(Layer layer : layers)//Just activates all the layers
			layer.activation(null, isInference);
		return linearNorm.getLastActivation();//Linear norm is the last layer in the stack, so its activation = this layer's activation.
	}

	@Override
	public void backprop() {
		linearNorm.backprop();
		linearResidual.backprop();
		linear.backprop();
		attentionNorm.backprop();
		attentionResidual.backprop();
		attention.backprop();
	}

	@Override
	public String name() {
		return "Encoder";
	}

	@Override
	public SimpleMatrix getLastActivation() {
		return linearNorm.getLastActivation();
	}
	
	@Override
	public void reportGradient(SimpleMatrix gradient) {
		/*
		 * This method needs to be overridden because if not, the layer after the encoder will send its gradients to the
		 * encoder layer, and the linearNorm will not receive the gradient because the next layer is connected to the
		 * encoder, not the linearNorm.
		 */
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public void setModel(LayersModel model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
	
	@Override
	public boolean[] getMasks() {
		return linearNorm.getMasks();
	}
	
	@Override
	public String stringify() {
		/*
		 * Returns a String with the form:
		 * thisId lastId attentionNormId
		 * attention.stringify()
		 * ##
		 * linear.stringify()
		 * ##
		 */
		StringBuilder builder = new StringBuilder();
		builder.append(getId() + " " + lastLayer.getId() + " "+ attentionNorm.getId() + "\n");
		builder.append(attention.stringify());
		builder.append("\n##\n");
		builder.append(linear.stringify());
		builder.append("\n##\n");
		return builder.toString();
	}
	
	/**
	 * Loads an Encoder based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static Encoder load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();//Gets this layer's id
		int lastID = scanner.nextInt();//Gets the preceding layer's id
		int normID = scanner.nextInt();//Gets its internal norm layer's id (linear will use getLayerByID to try to find this layer, so we need dto know it)
		scanner.useDelimiter("##");//Switches the delimiter to the one used in stringify()
		AttentionLayer attentionLayer = AttentionLayer.load(scanner.next(), model, position);
		model.reportLayer(attentionLayer);//Reports the attention layer to the model so that it can be found by model.getLayerByID()
		
		ResidualAddition residualAddition = new ResidualAddition(attentionLayer, model.getLayerByID(lastID));
		NormLayer norm = new NormLayer(residualAddition);
		norm.setId(normID);
		model.reportLayer(norm);//Reports the norm layer to the model so that it can be found by model.getLayerByID()
		
		StandardLayer linear = StandardLayer.load(scanner.next(), model, position);
		
		scanner.close();
		Encoder out = new Encoder(model.getLayerByID(lastID), attentionLayer, linear, residualAddition, norm);
		out.setId(id);
		return out;
	}
}
