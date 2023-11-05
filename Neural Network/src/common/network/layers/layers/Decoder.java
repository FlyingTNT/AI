package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersNetwork;

public class Decoder extends Layer {

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
		attention = new AttentionLayer(encoder, encoder, maskedAttentionNorm, heads, false, false);
		attentionResidual = new ResidualAddition(attention, maskedAttentionNorm);
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, last.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		
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
}
