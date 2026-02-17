# Deploy Seguro (Nginx + Auth + Rate Limit + HTTPS)

Este template adiciona uma camada de produção na frente do Streamlit:

- Autenticação básica (`nginx auth_basic`)
- Rate limiting por IP
- HTTPS com redirecionamento automático de HTTP para HTTPS
- Headers de segurança

## Arquivos criados

- `docker-compose.secure.yml`
- `deploy/nginx/nginx.conf`
- `deploy/nginx/conf.d/default.conf`
- `deploy/nginx/.htpasswd.example`
- `scripts/generate-nginx-htpasswd.ps1`

## 1) Preparar certificados TLS

Coloque os arquivos reais em `deploy/certs/`:

- `deploy/certs/fullchain.pem`
- `deploy/certs/privkey.pem`

## 2) Gerar usuário/senha do Nginx

No PowerShell:

```powershell
./scripts/generate-nginx-htpasswd.ps1 -Username admin -Password (Read-Host "Senha" -AsSecureString)
```

Isso criará:

- `deploy/nginx/.htpasswd`

## 3) Subir o ambiente seguro

Se estiver testando localmente e não tiver certificado ainda, você pode gerar um certificado self-signed:

```powershell
docker run --rm -v "${PWD}/deploy/certs:/certs" alpine/openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /certs/privkey.pem -out /certs/fullchain.pem -subj "/CN=localhost"
```

Depois suba os serviços:

```powershell
docker compose -f docker-compose.secure.yml up -d --build
```

## 4) Acessar

- `https://localhost`

Você será solicitado a autenticar com o usuário/senha do `.htpasswd`.

## Controles implementados

- **Auth básica**: `auth_basic` no Nginx
- **Rate limiting**: `10r/s` por IP, com burst `25`
- **Limite de conexões**: `30` por IP
- **Upload máximo**: `200m`
- **Headers**: HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy

## Observações de produção

- Use certificados válidos de CA confiável (Let's Encrypt, etc.).
- Troque `server_name _;` por seu domínio real.
- Não comite `deploy/nginx/.htpasswd` real no repositório.
- Para autenticação mais robusta, evolua para SSO/OIDC com proxy (ex.: oauth2-proxy).
