package common.network.layers.layers;

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
	
	public Decoder(Layer last, Encoder encoder, int heads, boolean isFirst)
	{
		super(last, last.outputs);
		maskedAttention = new AttentionLayer(last, last, last, heads, isFirst, isFirst);
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
	public float[][] activation(float[][] input) {
		for(Layer layer : layers)
			layer.activation(null);
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
	public void reportGradient(float[][] gradient) {
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public float[][] getLastActivation() {
		return linearNorm.getLastActivation();
	}
	
	@Override
	public void setModel(LayersNetwork model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
	
	public void setMasking(boolean masking)
	{
		maskedAttention.setMasking(masking);
	}
}
