package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersModel;

/**
 * A Decoder layer, as defined in Google's Attention is All You Need
 * @author C. Cooper
 */
public class Decoder extends Layer {
	
	Encoder encoder;//The encoder this decoder gets attention data from.
	
	AttentionLayer maskedAttention;//This decoder's masked attention layer
	ResidualAddition maskedAttentionResidual;//The masked attention's residual addition layer
	NormLayer maskedAttentionNorm;//The masked attention's norm layer
	AttentionLayer attention;//The cross-attention layer
	ResidualAddition attentionResidual;//The cross-attention's residual addition layer
	NormLayer attentionNorm;//The cross-attention's norm layer
	StandardLayer linear;//The decoder's linear layer
	ResidualAddition linearResidual;//The linear layer's residual addition layer
	NormLayer linearNorm;//The linear layer's norm layer
	
	Layer[] layers;//An array to hold all the layers (so I can use a for loop rather than typing them all out)
	
	/**
	 * Basic Decoder constructor.
	 * @param last The layer that precedes this one.
	 * @param encoder The encoder this layer performs cross-attention on.
	 * @param heads The number of heads of the attention layers.
	 * @param isFirst Whether this is the first decoder in the stack (the first needs to apply the casual mask).
	 * @param masking Whether to do masking.
	 */
	public Decoder(Layer last, Encoder encoder, int heads, boolean isFirst, boolean masking)
	{
		super(last, last.outputs);
		maskedAttention = new AttentionLayer(last, last, last, heads, isFirst, masking, true);//Creates the masked self attention layer (inputs = lastLayer)
		maskedAttentionResidual = new ResidualAddition(maskedAttention, last);//Creates the residual connection that bypasses the maskedAttention
		maskedAttentionNorm = new NormLayer(maskedAttentionResidual);
		attention = new AttentionLayer(encoder, encoder, maskedAttentionNorm, heads, false, masking, true);//Creates the cross-attention layer (k, v inputs = encoder, q = masked attention norm)
		attentionResidual = new ResidualAddition(attention, maskedAttentionNorm);//Creates the residual connection that bypasses the cross attention
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, last.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	/**
	 * More constructor used in {@link #load(String, LayersModel, int)}. Gives more control over the internal structure.
	 * @param last The last layer
	 * @param encoder The encoder this layer performs cross-attention on.
	 * @param maskedAttention The internal masked attention layer
	 * @param maskedResidualAddition The internal masked residual addition layer
	 * @param maskedNorm The internal masked norm layer
	 * @param attention The internal cross attention layer
	 * @param attentionResidual The internal cross attention residual layer 
	 * @param attentionNorm The internal cross attention norm layer
	 * @param linear The internal linear layer.
	 */
	private Decoder(Layer last, Encoder encoder, AttentionLayer maskedAttention, ResidualAddition maskedResidualAddition, NormLayer maskedNorm, AttentionLayer attention, ResidualAddition attentionResidual, NormLayer attentionNorm, StandardLayer linear)
	{
		super(last, last.outputs);
		this.maskedAttention = maskedAttention;
		this.maskedAttentionResidual = maskedResidualAddition;
		this.maskedAttentionNorm = maskedNorm;
		this.attention = attention;
		this.attentionResidual = attentionResidual;
		this.attentionNorm = attentionNorm;
		this.linear = linear;
		this.linearResidual = new ResidualAddition(linear, attentionNorm);
		this.linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		this.layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		for(Layer layer : layers)
<<<<<<< HEAD
			layer.activation(null);
=======
			layer.activation(null, isInference);
>>>>>>> refs/remotes/origin/ejml
		return linearNorm.getLastActivation();
	}

	@Override
	public void backprop() {
		linearNorm.backprop();
		linearResidual.backprop();
		linear.backprop();
		attentionNorm.backprop();
		attentionResidual.backprop();
		attention.backprop();
		maskedAttentionNorm.backprop();
		maskedAttentionResidual.backprop();
		maskedAttention.backprop();
	}

	@Override
	public String name() {
		return "Decoder";
	}

	@Override
	public void reportGradient(SimpleMatrix gradient) {
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public SimpleMatrix getLastActivation() {
		return linearNorm.getLastActivation();
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
		 * Returns a String in the form:
		 * thisId lastId encoderId maskedNormId attentionNormId
		 * maskedAttention.stringify()
		 * ##
		 * attention.stringify()
		 * ##
		 * linear.stringify()
		 * ##
		 */
		StringBuilder builder = new StringBuilder();
		builder.append(getId() + " " + lastLayer.getId() + " " + encoder.getId() + " " + maskedAttentionNorm.getId() + " " + attentionNorm.getId() + "\n");
		builder.append(maskedAttention.stringify());
		builder.append("\n##\n");
		builder.append(attention.stringify());
		builder.append("\n##\n");
		builder.append(linear.stringify());
		builder.append("\n##\n");
		return builder.toString();
	}

	/**
	 * Loads a Decoder based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static Decoder load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();//Gets the Decoder's id
		int lastID = scanner.nextInt();//Gets the last layer's id
		int encoderID = scanner.nextInt();//Gets the decoder's encoder's id
		int maskedNormID = scanner.nextInt();//Gets the decoder's masked norm's id
		int normID = scanner.nextInt();//Gets the decoder's cross attention norm's id
		scanner.useDelimiter("##");//Sets the scanner delimiter to "##" (Used to seperate the internal layers in stringify())
		AttentionLayer maskedAttentionLayer = AttentionLayer.load(scanner.next(), model, position);//Loads the masked attention layer
		model.reportLayer(maskedAttentionLayer);//Reports the masked attention to the model so that it can be found by model.getLayerByID()
		
		ResidualAddition maskedResidualAddition = new ResidualAddition(maskedAttentionLayer, model.getLayerByID(lastID));
		NormLayer maskedNorm = new NormLayer(maskedResidualAddition);
		maskedNorm.setId(maskedNormID);
		model.reportLayer(maskedNorm);//Reports the masked norm to the model so that it can be found by model.getLayerByID()
		
		AttentionLayer attentionLayer = AttentionLayer.load(scanner.next(), model, position);
		model.reportLayer(attentionLayer);//Reports the cross attention to the model so that it can be found by model.getLayerByID()
		
		ResidualAddition residualAddition = new ResidualAddition(attentionLayer, maskedNorm);
		NormLayer norm = new NormLayer(residualAddition);
		norm.setId(normID);
		model.reportLayer(norm);//Reports the cross attention norm to the model so that it can be found by model.getLayerByID()
		
		StandardLayer linear = StandardLayer.load(scanner.next(), model, position);
		scanner.close();
		Decoder out = new Decoder(model.getLayerByID(lastID), (Encoder)model.getLayerByID(encoderID), maskedAttentionLayer, maskedResidualAddition, maskedNorm, attentionLayer, residualAddition, norm, linear);
		out.setId(id);
		return out;
	}
}
