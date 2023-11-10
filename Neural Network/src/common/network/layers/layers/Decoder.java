package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersNetwork;

public class Decoder extends Layer {

	Encoder encoder;
	
	AttentionLayer maskedAttention;
	ResidualAddition maskedAttentionResidual;
	NormLayer maskedAttentionNorm;
	AttentionLayer attention;
	ResidualAddition attentionResidual;
	NormLayer attentionNorm;
	StandardLayer linear;
	ResidualAddition linearResidual;
	NormLayer linearNorm;
	
	Layer[] layers;
	
	public Decoder(Layer last, Encoder encoder, int heads, boolean isFirst, boolean masking)
	{
		super(last, last.outputs);
		maskedAttention = new AttentionLayer(last, last, last, heads, masking, isFirst);
		maskedAttentionResidual = new ResidualAddition(maskedAttention, last);
		maskedAttentionNorm = new NormLayer(maskedAttentionResidual);
		attention = new AttentionLayer(encoder, encoder, maskedAttentionNorm, heads, masking, false);
		attentionResidual = new ResidualAddition(attention, maskedAttentionNorm);
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, last.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	private Decoder(Layer last, Encoder encoder, AttentionLayer maskedAttention, AttentionLayer attention, StandardLayer linear)
	{
		super(last, last.outputs);
		this.maskedAttention = maskedAttention;
		maskedAttentionResidual = new ResidualAddition(maskedAttention, last);
		maskedAttentionNorm = new NormLayer(maskedAttentionResidual);
		this.attention = attention;
		attentionResidual = new ResidualAddition(attention, maskedAttentionNorm);
		attentionNorm = new NormLayer(attentionResidual);
		this.linear = linear;
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		this.encoder = encoder;
		layers = new Layer[]{maskedAttention, maskedAttentionResidual, maskedAttentionNorm,
									 attention, attentionResidual, attentionNorm,
									 linear, linearResidual, linearNorm};
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		for(Layer layer : layers)
			layer.activation(null);
		return linearNorm.lastActivation;
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
	public void setModel(LayersNetwork model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
	
	@Override
	public boolean[] getMasks() {
		return linearNorm.getMasks();
	}
	
	public void setMasking(boolean masking)
	{
		maskedAttention.setMasking(masking);
		attention.setMasking(masking);
	}
	
	public void setInference(boolean inf)
	{
		maskedAttention.decoder = !inf;
	}
	
	@Override
	public String stringify() {
		StringBuilder builder = new StringBuilder();
		builder.append(getId() + " " + lastLayer.getId() + " " + encoder.getId() + "\n");
		builder.append(maskedAttention.stringify());
		builder.append("\n##\n");
		builder.append(attention.stringify());
		builder.append("\n##\n");
		builder.append(linear.stringify());
		builder.append("\n##\n");
		return builder.toString();
	}
	
	@Override
	public Decoder load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		int encoderID = scanner.nextInt();
		scanner.useDelimiter("##");
		AttentionLayer builder = new AttentionLayer();
		AttentionLayer maskedAttentionLayer = builder.load(scanner.next(), model, position);
		AttentionLayer attentionLayer = builder.load(scanner.next(), model, position);
		model.reportLayer(attentionLayer);
		StandardLayer linear = attentionLayer.keyLinear.load(scanner.next(), model, position);
		scanner.close();
		Decoder out = new Decoder(model.getLayerByID(lastID), (Encoder)model.getLayerByID(encoderID), maskedAttentionLayer, attentionLayer, linear);
		out.setId(id);
		return out;
	}
}
