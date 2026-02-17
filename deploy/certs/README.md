# Certificados TLS

Coloque aqui os arquivos usados pelo Nginx:

- `fullchain.pem`
- `privkey.pem`

Para ambiente local (desenvolvimento), você pode gerar um certificado self-signed com Docker:

```powershell
docker run --rm -v "${PWD}/deploy/certs:/certs" alpine/openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /certs/privkey.pem -out /certs/fullchain.pem -subj "/CN=localhost"
```

> Em produção, use certificado emitido por uma CA confiável (ex.: Let's Encrypt).
