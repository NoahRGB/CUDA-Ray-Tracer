#version 430

in vec3 in_Position;
in vec2 in_TextureCoord; 

out vec4 ex_Color;
out vec2 ex_TextureCoord;

void main(void) {
	ex_TextureCoord = in_TextureCoord;
	gl_Position = vec4(in_Position.x, in_Position.y, in_Position.z, 1.0);
}