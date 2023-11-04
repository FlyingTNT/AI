package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.Activation;
import common.network.layers.models.LayersNetwork;

public class Encoder extends Layer{

	AttentionLayer attention;
	ResidualAddition attentionResidual;
	NormLayer attentionNorm;
	StandardLayer linear;
	ResidualAddition linearResidual;
	NormLayer linearNorm;
	
	Layer[] layers;

	public Encoder(Layer lastLayer, int heads) {
		super(lastLayer, lastLayer.outputs);
		this.depth = lastLayer.depth;
		
		attention = new AttentionLayer(lastLayer, lastLayer, lastLayer, heads, false, false);
		attentionResidual = new ResidualAddition(attention, lastLayer);
		attentionNorm = new NormLayer(attentionResidual);
		linear = new StandardLayer(attentionNorm, lastLayer.outputs, Activation.RELU);
		linearResidual = new ResidualAddition(linear, attentionNorm);
		linearNorm = new NormLayer(linearResidual);
		
		layers = new Layer[] {attention, attentionResidual, attentionNorm, linear, linearResidual, linearNorm};
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		for(Layer layer : layers)
			layer.activation(null);
		return lastActivation;
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
		linearNorm.reportGradient(gradient);
	}
	
	@Override
	public void setModel(LayersNetwork model) {
		super.setModel(model);
		for(Layer layer : layers)
			layer.setModel(model);
	}
}
