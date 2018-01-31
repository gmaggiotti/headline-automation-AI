# -*- coding: utf-8 -*-
import re

import HTMLParser
def polish_sentence( sentence ):
    p = HTMLParser.HTMLParser()
    sentence = p.unescape(unicode(sentence, "utf-8"))
    sentence = re.sub(u'\n','', sentence)
    sentence = re.sub(u'<[^>]*>nt','', sentence)
    sentence = re.sub(u'<[^>]*>','', sentence)
    sentence = re.sub(u'\[[a-z\_]*embed:.*\]','', sentence)
    sentence = re.sub(u'\[video:.*\]','', sentence)
    sentence = re.sub(u'[\.\[\]\?\,\(\)\!\"\'\\/\:\-]',' ', sentence)
    sentence = re.sub(u'[ ]+',' ', sentence)
    sentence = re.sub(u'%[0-9][a-zA-Z-0-9]', ' ',sentence)
    return sentence

str = '<p>ntEl DT de <a href="http://www.bocajuniors.com.ar/home/sitio">Boca</a>, <a href="http://www.tn.com.ar/tags/julio-falcioni">Julio C&eacute;sar</a><strong><a href="http://www.tn.com.ar/tags/julio-falcioni"> Falcioni</a>, asegur&oacute; que si <a href="http://www.tn.com.ar/tags/juan-roman-riquelme">Rom&aacute;n Riquelme</a> "est&aacute; bien f&iacute;sicamente, va a ser fundamental" </strong>para el equipo.</p>'


print polish_sentence(str)