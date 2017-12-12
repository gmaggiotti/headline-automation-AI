# -*- coding: utf-8 -*-
import re

def polish_sentence( sentence ):
    spcl_chr = re.escape('[]?,()!"\'\\/:')
    regex = '[' + spcl_chr + ']'
    regex_array = ['<a[^>]*>','</a>','<span[^>]*>','</span>','<p[^>]*>','</p>','<strong[^>]*>','</strong>',regex,"&rdquo;","<blockquote>", "%20" ]
    sentence = re.sub(u'&ntilde;', u"ñ",sentence)
    sentence = re.sub(u'[a-zA-Z]\.', " . ",sentence)
    for i in range(regex_array.__len__()):
        sentence = re.sub(regex_array[i]," ", sentence)
    return sentence

str = u'Cagna, sincero: "Supongo que si no llegan los resultados llegará el despido" "Supongo que, como todos, dependo de los resultados, y si no llegan, llegará el despido", dijo con brutal sinceridad, <a href="http://www.tn.com.ar/personajes/diego-cagna">Diego Cagna</a>, el técnico de un <a href="http://www.tn.com.ar/tags/estudiantes-de-la-plata">Estudiantes de La Plata </a>que perdió los dos primeros partidos del torneo Final 2013 (también perdió el último del torneo Inicial).En conferencia de prensa, el entrenador se&ntilde;aló que "hay veces que decido no hablar porque igual muchos ponen lo que quieren". Y advirtió: "Si me quieren echar, seguramente me lo dirán. Yo no voy a renunciar y si me toca irme lo haré muy tranquilo" .Cagna le hizo frente a los rumores que circulan en La Plata, y manifestó: "Muchos de ustedes (por los periodistas) tanto en la Capital Federal como acá en La Plata, mencionan a Matías Almeyda, a Gabriel Milito. Tiran nombres, pero no me molesta, me acostumbré. No se por qué pasa, pero pasa".Al referirse al nivel del equipo, el DT aseguró que "hasta ahora es un poco lo mismo del campeonato pasado: somos irregulares, nos falta más juego y la verdad es que no podemos tener tantos altibajos".Sobre la impaciencia de los hinchas, en cambio, se mostró comprensivo: "Ellos quieren ganar y nosotros también. Solo le podemos pedir disculpas por no haberle dado alegrías hasta ahora".El Pincha jugará ma&ntilde;ana a las 21.15 contra <a href="http://www.tn.com.ar/tags/san-lorenzo">San Lorenzo</a>, en el Ciudad de La Plata, con la necesidad de ganar por primera vez en el certamen%3B%20Gabriel%20Anello%20.'

print polish_sentence(str)