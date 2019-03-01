---
title: "KL Divergence"
date: 2019-03-01
classes: wide
use_math: true
tags: reinforcement_learning tensorflow theano KL divegence
category: reinforcement learning
---


[KL](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)
[KL](https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians)

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>N</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>&#x03BC;<!-- μ --></mi>
    <mn>1</mn>
  </msub>
  <mo>,</mo>
  <msub>
    <mi>&#x03C3;<!-- σ --></mi>
    <mn>1</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math>

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>q</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>N</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>&#x03BC;<!-- μ --></mi>
    <mn>2</mn>
  </msub>
  <mo>,</mo>
  <msub>
    <mi>&#x03C3;<!-- σ --></mi>
    <mn>2</mn>
  </msub>
  <mo stretchy="false">)</mo>
</math>


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true">
    <mtr>
      <mtd>
        <mi>K</mi>
        <mi>L</mi>
        <mo stretchy="false">(</mo>
        <mi>p</mi>
        <mo>,</mo>
        <mi>q</mi>
        <mo stretchy="false">)</mo>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <mo>&#x2212;<!-- − --></mo>
        <mo>&#x222B;<!-- ∫ --></mo>
        <mi>p</mi>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mi>q</mi>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mi>d</mi>
        <mi>x</mi>
        <mo>+</mo>
        <mo>&#x222B;<!-- ∫ --></mo>
        <mi>p</mi>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mi>p</mi>
        <mo stretchy="false">(</mo>
        <mi>x</mi>
        <mo stretchy="false">)</mo>
        <mi>d</mi>
        <mi>x</mi>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <mfrac>
          <mn>1</mn>
          <mn>2</mn>
        </mfrac>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mo stretchy="false">(</mo>
        <mn>2</mn>
        <mi>&#x03C0;<!-- π --></mi>
        <msubsup>
          <mi>&#x03C3;<!-- σ --></mi>
          <mn>2</mn>
          <mn>2</mn>
        </msubsup>
        <mo stretchy="false">)</mo>
        <mo>+</mo>
        <mfrac>
          <mrow>
            <msubsup>
              <mi>&#x03C3;<!-- σ --></mi>
              <mn>1</mn>
              <mn>2</mn>
            </msubsup>
            <mo>+</mo>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>&#x03BC;<!-- μ --></mi>
              <mn>1</mn>
            </msub>
            <mo>&#x2212;<!-- − --></mo>
            <msub>
              <mi>&#x03BC;<!-- μ --></mi>
              <mn>2</mn>
            </msub>
            <msup>
              <mo stretchy="false">)</mo>
              <mn>2</mn>
            </msup>
          </mrow>
          <mrow>
            <mn>2</mn>
            <msubsup>
              <mi>&#x03C3;<!-- σ --></mi>
              <mn>2</mn>
              <mn>2</mn>
            </msubsup>
          </mrow>
        </mfrac>
        <mo>&#x2212;<!-- − --></mo>
        <mfrac>
          <mn>1</mn>
          <mn>2</mn>
        </mfrac>
        <mo stretchy="false">(</mo>
        <mn>1</mn>
        <mo>+</mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mn>2</mn>
        <mi>&#x03C0;<!-- π --></mi>
        <msubsup>
          <mi>&#x03C3;<!-- σ --></mi>
          <mn>1</mn>
          <mn>2</mn>
        </msubsup>
        <mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd />
    </mtr>
    <mtr>
      <mtd />
      <mtd>
        <mi></mi>
        <mo>=</mo>
        <mi>log</mi>
        <mo>&#x2061;<!-- ⁡ --></mo>
        <mfrac>
          <msub>
            <mi>&#x03C3;<!-- σ --></mi>
            <mn>2</mn>
          </msub>
          <msub>
            <mi>&#x03C3;<!-- σ --></mi>
            <mn>1</mn>
          </msub>
        </mfrac>
        <mo>+</mo>
        <mfrac>
          <mrow>
            <msubsup>
              <mi>&#x03C3;<!-- σ --></mi>
              <mn>1</mn>
              <mn>2</mn>
            </msubsup>
            <mo>+</mo>
            <mo stretchy="false">(</mo>
            <msub>
              <mi>&#x03BC;<!-- μ --></mi>
              <mn>1</mn>
            </msub>
            <mo>&#x2212;<!-- − --></mo>
            <msub>
              <mi>&#x03BC;<!-- μ --></mi>
              <mn>2</mn>
            </msub>
            <msup>
              <mo stretchy="false">)</mo>
              <mn>2</mn>
            </msup>
          </mrow>
          <mrow>
            <mn>2</mn>
            <msubsup>
              <mi>&#x03C3;<!-- σ --></mi>
              <mn>2</mn>
              <mn>2</mn>
            </msubsup>
          </mrow>
        </mfrac>
        <mo>.</mo>
      </mtd>
    </mtr>
  </mtable>
</math>
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>K</mi>
  <mi>L</mi>
  <mo stretchy="false">(</mo>
  <mi>p</mi>
  <mo>,</mo>
  <mi>q</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mfrac>
    <msub>
      <mi>&#x03C3;<!-- σ --></mi>
      <mn>2</mn>
    </msub>
    <msub>
      <mi>&#x03C3;<!-- σ --></mi>
      <mn>1</mn>
    </msub>
  </mfrac>
  <mo>+</mo>
  <mfrac>
    <mrow>
      <msubsup>
        <mi>&#x03C3;<!-- σ --></mi>
        <mn>1</mn>
        <mn>2</mn>
      </msubsup>
      <mo>+</mo>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>&#x03BC;<!-- μ --></mi>
        <mn>1</mn>
      </msub>
      <mo>&#x2212;<!-- − --></mo>
      <msub>
        <mi>&#x03BC;<!-- μ --></mi>
        <mn>2</mn>
      </msub>
      <msup>
        <mo stretchy="false">)</mo>
        <mn>2</mn>
      </msup>
    </mrow>
    <mrow>
      <mn>2</mn>
      <msubsup>
        <mi>&#x03C3;<!-- σ --></mi>
        <mn>2</mn>
        <mn>2</mn>
      </msubsup>
    </mrow>
  </mfrac>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
</math>