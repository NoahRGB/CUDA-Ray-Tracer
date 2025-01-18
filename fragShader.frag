#version 430

in vec4 ex_Color;
in vec2 ex_TextureCoord;

uniform sampler2D DiffuseMap;

out vec4 out_Color; // final colour for pixel

void main(void) {
	out_Color = texture(DiffuseMap, ex_TextureCoord);
}