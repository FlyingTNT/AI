package common.network.layers;

import common.network.layers.models.TransformerModel2D;

public class TransMain4 {
	static int TOKEN_EMBED_DEPTH = 128;
	static int TYPE_EMBED_DEPTH = 128;
	static int EMBED_DEPTH = TOKEN_EMBED_DEPTH + TYPE_EMBED_DEPTH;
	static int HEADS = 4;
	static int TRANSFORMER_STACK_SIZE = 6;
	static int ENCODER_VOCAB_SIZE = 839;
	static int ENCODER_SEQUENCE_LENGTH = 1378;
	static int DECODER_VOCAB_SIZE = 339;
	static int DECODER_TYPE_SIZE = 0;
	static int DECODER_SEQUENCE_LENGTH = 716;
	static float LEARNING_RATE = 0.0005f;
	
	public static void main(String[] args) {
		TransformerModel2D model = new TransformerModel2D(LEARNING_RATE, ENCODER_SEQUENCE_LENGTH, DECODER_SEQUENCE_LENGTH, null, new int[] {TOKEN_EMBED_DEPTH, TYPE_EMBED_DEPTH}, EMBED_DEPTH, new int[] {ENCODER_VOCAB_SIZE}, new int[] {DECODER_VOCAB_SIZE, DECODER_TYPE_SIZE}, HEADS, 6);
		
	}

}
